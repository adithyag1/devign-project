import json
import re
import subprocess
import os.path
import os
import time
from .cpg_client_wrapper import CPGClientWrapper
#from ..data import datamanager as data


def funcs_to_graphs(funcs_path):
    client = CPGClientWrapper()
    # query the cpg for the dataset
    print(f"Creating CPG.")
    graphs_string = client(funcs_path)
    # removes unnecessary namespace for object references
    graphs_string = re.sub(r"io\.shiftleft\.codepropertygraph\.generated\.", '', graphs_string)
    graphs_json = json.loads(graphs_string)

    return graphs_json["functions"]

"""
def graph_indexing(graph):
    idx = int(graph["file"].split(".c")[0].split("/")[-1])
    del graph["file"]
    return idx, {"functions": [graph]}
"""
#changed
def graph_indexing(graph):
    try:
        file_path = graph["file"]
        file_name = file_path.split("/")[-1]
        file_id = file_name.split(".c")[0]
        idx = int(file_id)
        del graph["file"]
        return idx, {"functions": [graph]}
    except:
        return None, None  

def joern_parse(joern_path, input_path, output_path, file_name):
    out_file = file_name + ".bin"
    
    # 1. ENSURE THE DIRECTORY EXISTS
    # This was the cause of your "nio:/.../data/cpg" error
    if not os.path.exists(output_path):
        os.makedirs(output_path, exist_ok=True)
    
    abs_input = os.path.abspath(input_path)
    abs_output = os.path.abspath(os.path.join(output_path, out_file))
    executable = os.path.join(joern_path, "joern-parse")
    
    joern_parse_call = subprocess.run(
        [executable, abs_input, "--output", abs_output], 
        stdout=subprocess.PIPE, 
        stderr=subprocess.PIPE,
        text=True,
        cwd=joern_path
    )
    
    if joern_parse_call.returncode != 0:
        print("\n" + "!"*20 + " JOERN PARSE ERROR " + "!"*20)
        print(f"Stdout: {joern_parse_call.stdout}")
        print(f"Stderr: {joern_parse_call.stderr}")
        joern_parse_call.check_returncode()
        
    return out_file

def joern_create(joern_path, in_path, out_path, cpg_files):
    json_files = []
    # Resolve the devign_project root from src/prepare/
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    joern_bin = os.path.join(project_root, "joern", "joern-cli", "joern")
    # Updated to the singular filename
    base_script = os.path.join(project_root, "joern", "graph-for-funcs.sc")
    
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    for cpg_file in cpg_files:
        json_file_name = f"{cpg_file.replace('.bin', '')}.json"
        json_files.append(json_file_name)
        
        cpg_full_path = os.path.abspath(os.path.join(in_path, cpg_file))
        json_out_path = os.path.abspath(os.path.join(out_path, json_file_name))
        
        if os.path.exists(cpg_full_path):
            print(f"Exporting JSON for {cpg_file}...")
            
            # The Scala script writes this to the current working directory (project root)
            script_out_name = os.path.join(os.getcwd(), "last_graph_export.json")
            if os.path.exists(script_out_name):
                os.remove(script_out_name)

            tmp_script_path = os.path.abspath("tmp_export.sc")
            with open(tmp_script_path, "w") as f:
                f.write(f'importCpg("{cpg_full_path}")\n')
                f.write(f'runScript("{base_script}", Map.empty[String, String], cpg)\n')
                f.write(f'delete\n')

            try:
                # Set Java heap to 10GB for memory-intensive graph exports
                custom_env = os.environ.copy()
                custom_env["JAVA_OPTS"] = "-Xmx10g"

                subprocess.run(
                    [joern_bin, "--script", tmp_script_path],
                    check=True, capture_output=True, text=True,
                    env=custom_env
                )
                
                # Move the file to data/cpg/ once generated
                if os.path.exists(script_out_name):
                    os.rename(script_out_name, json_out_path)
                    print(f"Successfully created {json_file_name}")
                else:
                    print(f"ERROR: Scala script failed to produce {script_out_name}")
            
            except subprocess.CalledProcessError as e:
                print(f"JOERN EXECUTION FAILED: {e.stderr}")
            finally:
                if os.path.exists(tmp_script_path):
                    os.remove(tmp_script_path)
                
    return json_files

def json_process(in_path, json_file):
    """
    Extract ONE function per source file - the REAL function, not synthetic wrappers.
    
    Pattern found: Each file has 2 entries:
    - {file}:<global>  (synthetic, skip this)
    - actual_function_name (keep this)
    """
    file_full_path = os.path.join(in_path, json_file)
    if os.path.exists(file_full_path):
        with open(file_full_path) as jf:
            cpg_string = jf.read().strip()
            if not cpg_string:
                print(f"Warning: {json_file} is empty.")
                return None
            cpg_string = re.sub(r"io\.shiftleft\.codepropertygraph\.generated\.", '', cpg_string)
            try:
                cpg_json = json.loads(cpg_string)
                
                # GROUP functions by source file
                functions_by_file = {}
                for graph in cpg_json["functions"]:
                    if graph["file"] != "N/A" and graph["file"] != "<empty>":
                        file_path = graph["file"]
                        file_id = file_path.split(".c")[0].split("/")[-1]
                        try:
                            idx = int(file_id)
                            func_name = graph.get("function", "")
                            
                            # SKIP synthetic functions
                            if "<global>" in func_name or func_name == "START_TEST" or func_name.startswith("<"):
                                continue
                            
                            # Keep ONLY the first real function per file
                            # (There should be exactly one after filtering)
                            if idx not in functions_by_file:
                                functions_by_file[idx] = graph
                        except:
                            pass
                
                container = [(idx, {"functions": [graph]}) 
                            for idx, graph in sorted(functions_by_file.items())]
                
                return container if container else None
                
            except json.JSONDecodeError:
                print(f"Error: Failed to decode JSON in {json_file}")
                return None
    return None
'''
def generate(dataset, funcs_path):
    dataset_size = len(dataset)
    print("Size: ", dataset_size)
    graphs = funcs_to_graphs(funcs_path[2:])
    print(f"Processing CPG.")
    container = [graph_indexing(graph) for graph in graphs["functions"] if graph["file"] != "N/A"]
    graph_dataset = data.create_with_index(container, ["Index", "cpg"])
    print(f"Dataset processed.")

    return data.inner_join_by_index(dataset, graph_dataset)
'''

# client = CPGClientWrapper()
# client.create_cpg("../../data/joern/")
# joern_parse("../../joern/joern-cli/", "../../data/joern/", "../../joern/joern-cli/", "gen_test")
# print(funcs_to_graphs("/data/joern/"))
"""
while True:
    raw = input("query: ")
    response = client.query(raw)
    print(response)
"""
