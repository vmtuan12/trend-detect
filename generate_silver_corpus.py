import os
import numpy as np
import networkx as nx
from collections import defaultdict
import json
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('xlm-roberta-large')

def list_files(directory):
    """Returns a list of files in the given directory."""
    try:
        return [f for f in os.listdir(directory) if (os.path.isfile(os.path.join(directory, f)))]
    except FileNotFoundError:
        print(f"Directory '{directory}' not found.")
        return []
    except PermissionError:
        print(f"Permission denied for directory '{directory}'.")
        return []

def build_event_graph(json_list):
    G = nx.DiGraph()
    event_counter = defaultdict(int)
    event_arguments = defaultdict(lambda: defaultdict(int))
    event_type_count = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(int))))
    
    for data in json_list:
        entity_map = {entity["entity_id"]: entity["entity_type"] for entity in data.get("entity_mentions", [])}
        
        for trigger in data.get("event_triggers", []):
            event_type = trigger["event_type"]
            trigger_id = trigger["trigger_id"]
            
            # Increment event count
            event_counter[event_type] += 1
            
            # Collect arguments
            roles = []
            type_count = defaultdict(lambda: defaultdict(int))
            
            for arg in data.get("event_arguments", []):
                if arg["trigger_id"] == trigger_id:
                    role = arg["role_type"]
                    entity_type = entity_map.get(arg["entity_id"], "UNKNOWN")
                    roles.append(role)
                    type_count[role][entity_type] += 1
            
            if roles:
                roles_list = list(roles)  # Ensure list instead of tuple
                event_arguments[event_type][tuple(roles_list)] += 1  # Convert back to tuple for dict key
                
                if tuple(roles_list) not in event_type_count[event_type]:
                    event_type_count[event_type][tuple(roles_list)] = {}
                
                for role, entity_counts in type_count.items():
                    if role not in event_type_count[event_type][tuple(roles_list)]:
                        event_type_count[event_type][tuple(roles_list)][role] = {}
                    for entity_type, count in entity_counts.items():
                        if entity_type not in event_type_count[event_type][tuple(roles_list)][role]:
                            event_type_count[event_type][tuple(roles_list)][role][entity_type] = 1
                        else:
                            event_type_count[event_type][tuple(roles_list)][role][entity_type] += count
    
    sorted_events = sorted(event_counter.items(), key=lambda x: x[1], reverse=True)
    
    # Add nodes to the graph
    for event_type, count in sorted_events:
        arguments_list = [
            {
                "value": list(key),  # Convert tuple to list here
                "count": value,
                "type_count": {role: dict(entity_counts) for role, entity_counts in event_type_count[event_type][key].items()}
            } 
            for key, value in event_arguments[event_type].items()
        ]
        G.add_node(event_type, count=count, arguments=arguments_list)
    
    return G

def argument_threshold(arr, divide_coefficient=3) -> float | int:
    min_num = max(arr)
    max_num = min(arr)

    q1 = np.percentile(arr, 25)
    q3 = np.percentile(arr, 75)
    iqr = q3 - q1
    
    if iqr <= max_num / divide_coefficient:
        return 0
    
    threshold = min_num + (max_num - min_num)/2
    return threshold

def sort_event_dict(event_dict):
    empty_types = []
    for event_type in event_dict:
        if len([arg['count'] for arg in event_dict[event_type]['arguments']]) == 0:
            empty_types.append(event_type)

    for t in empty_types:
        event_dict.pop(t)
        
    for event_type in event_dict:
        try:
            threshold = argument_threshold([arg['count'] for arg in event_dict[event_type]['arguments']])
        except Exception as e:
            print(event_dict)
            print([arg['count'] for arg in event_dict[event_type]['arguments']])
            print(event_type)
            # print(e)
            raise(e)
        event_dict[event_type]['arguments'].sort(key=lambda x: x['count'], reverse=True)
        event_dict[event_type]['arguments'] = [arg for arg in event_dict[event_type]['arguments'] if arg['count'] >= threshold]

def filter_records(records, result_graph) -> dict:
    matched_records = dict()
    
    for record in records:
        for event in record.get("event_triggers", []):
            event_type = event["event_type"]
            if event_type not in matched_records:
                matched_records[event_type] = []
            
            if event_type in result_graph:
                event_args = sorted([arg["role_type"] for arg in record.get("event_arguments", []) 
                                     if arg["trigger_id"] == event["trigger_id"]])
                
                for arg_entry in result_graph[event_type]["arguments"]:
                    expected_args = sorted(arg_entry["value"])
                    
                    if event_args == expected_args:
                        matched_records[event_type].append(record)
                        break
    
    list_types = list(matched_records.keys())
    for k in list_types:
        if len(matched_records[k]) == 0:
            matched_records.pop(k)

    return matched_records

def convert_json(input_json):
    tokens = input_json["tokens"]
    group_pieces = [[p for p in tokenizer.tokenize(w) if p != '▁'] for w in tokens]
    for ps in group_pieces:
        if len(ps) == 0:
            ps += ['-']
    pieces = [p for ps in group_pieces for p in ps]
    token_lens = [len(x) for x in group_pieces]

    entity_id_map = {}
    
    entity_mentions = []
    for index, entity in enumerate(input_json["entity_mentions"]):
        entity_id = f"train-{input_json['sent_id']}-T{index + 1}"
        entity_id_map[entity["entity_id"]] = entity_id
        entity_mentions.append({
            "id": entity_id,
            "text": entity["text"],
            "entity_type": entity["entity_type"],
            "start": entity["start_token"],
            "end": entity["end_token"],
            "start_char": input_json["text"].find(entity["text"]),
            "end_char": input_json["text"].find(entity["text"]) + len(entity["text"]),
            "mention_type": "UNK"
        })
    
    event_mentions = []
    for index, event in enumerate(input_json["event_triggers"]):
        event_id = f"train-{input_json['sent_id']}-T{index + 10}"
        arguments = [
            {
                "entity_id": entity_id_map[arg["entity_id"]],
                "text": next(e["text"] for e in input_json["entity_mentions"] if e["entity_id"] == arg["entity_id"]),
                "role": arg["role_type"]
            }
            for arg in input_json["event_arguments"] if arg["trigger_id"] == event["trigger_id"]
        ]
        event_mentions.append({
            "id": event_id,
            "event_type": event["event_type"],
            "trigger": {
                "text": event["text"],
                "start": event["start_token"],
                "end": event["end_token"],
                "start_char": input_json["text"].find(event["text"]),
                "end_char": input_json["text"].find(event["text"]) + len(event["text"]),
            },
            "arguments": arguments
        })
    
    output_json = {
        "doc_id": f"train-{input_json['sent_id']}",
        "sent_id": f"train-{input_json['sent_id']}",
        "tokens": input_json["tokens"],
        "sentence": input_json["text"],
        "url": input_json["url"],
        "date": input_json["date"],
        "pieces": pieces,
        "token_lens": token_lens,
        "entity_mentions": entity_mentions,
        "event_mentions": event_mentions,
        "relation_mentions": []
    }
    
    return output_json

final_data = dict()
data = []
count_event_exists = 0
file_path = "/home/mhtuan/work/mbf/result/filtered_matched_date.json"
final_data[file_path] = dict()
with open(file_path, 'r') as f:
    lines = f.readlines()
    for line in lines:
        json_rec = json.loads(line)
        data.append(json_rec)
        if len(json_rec['event_triggers']) > 0:
            count_event_exists += 1

print(f"{count_event_exists} events detected within {len(data)} sentences")

G = build_event_graph(data)
for node in G.nodes(data=True):
    # if node[1]['count'] > (count_event_exists / 10):
    #     # print(node)
    final_data[file_path][node[0]] = node[1]

# print(total_all, total_events)
# exit(0)

with open("graph_event.json", "w+", encoding='utf-8') as ff:
    json.dump(final_data, ff, ensure_ascii=False, indent=4)

# with open("graph_event.json") as ff:
#     result_graph_data = json.load(ff)

# final_result = dict()
# for file_path in result_graph_data:
#     with open(file_path) as ff:
#         records = []
#         for line in ff.readlines():
#             records.append(json.loads(line.strip()))
            
#     sort_event_dict(result_graph_data[file_path])

#     filtered_records = filter_records(records, result_graph_data[file_path])
#     for k in filtered_records.keys():
#         for item in filtered_records[k]:
#             final_result[item['sent_id']] = item

# for index, item in enumerate(final_result.values()):
#     item['sent_id'] = index + 100000

# with open("result.json", "w+") as f:
#     for input_json in final_result.values():
#         # input_json['sent_id'] = input_json['sent_id']
#         output_json = convert_json(input_json)
#         f.write(json.dumps(output_json, ensure_ascii=False) + "\n")