# HEPData yoda, yaml to adl efficiency table converter
# Author: Ekin Sıla Yörük

import os
import re
import yaml
import requests

def process_yaml_data(yaml_content):
    data = yaml.safe_load(yaml_content)

    # Extract dependent variable (Efficiency or event count) and its name from yaml 
    dependent_variable_name = data["dependent_variables"][0]["header"]["name"].replace(" ", "_") # Use yaml-specified name & replace spaces 
    table_name = f"{dependent_variable_name}_table" # Format table name and type
    table_type = dependent_variable_name

    efficiency_values = [] 
    err_minus = [] 
    err_plus = [] 

    for entry in data["dependent_variables"][0]["values"]: 
        efficiency_values.append(entry["value"])

        #Checking errors 
        if "errors" in entry:
            error_entry = entry["errors"][0]
            if "asymerror" in error_entry:
                err_minus.append(abs(error_entry["asymerror"]["minus"]))
                err_plus.append(error_entry["asymerror"]["plus"])

            elif "symerror" in error_entry:
                err_minus.append(error_entry["symerror"])
                err_plus.append(error_entry["symerror"])

            else:
                err_minus.append(0.0)
                err_plus.append(0.0)
    
        else:
            err_minus.append(0.0)
            err_plus.append(0.0)

    # Extract independent variables 
    independent_variables = [] 
    variable_names = [] 
    for var in data["independent_variables"]: 
        variable_names.append(var["header"]["name"]) # Store variable names 
        independent_variables.append([entry["value"] for entry in var["values"]]) # Store values 
    
    # Generate min values for each independent variable 
    min_values = [] 
    for var_values in independent_variables: 
        min_values.append([0] + var_values[:-1]) # First min is 0, others are previous max 
    

    # Write to an ADL file
    adl_filename = "output.adl"

    with open(adl_filename, "w") as adl_file:
        adl_file.write(f"table {table_name}\n")  # Use custom table name
        adl_file.write(f"tabletype {table_type}\n")
        adl_file.write(f"nvars {len(independent_variables)}\n")  # Number of independent variables
        adl_file.write("errors true\n")

        # Writing header with the correct dependent variable name
        header = f"# {dependent_variable_name:<12} {'err-':<12} {'err+':<12} " + " ".join(
            [f"{var}_Min".ljust(15) + f"{var}_Max".ljust(15) for var in variable_names]) + "\n"
        adl_file.write(header)

        # Writing data with aligned columns
        for i in range(len(efficiency_values)):
            row = f"{efficiency_values[i]:<12.6f} {err_minus[i]:<12.4f} {err_plus[i]:<12.4f}"
            for j in range(len(independent_variables)):
                row += f" {min_values[j][i]:<15.3f} {independent_variables[j][i]:<15.3f}"
            adl_file.write(row + "\n")

    print(f"ADL file '{adl_filename}' has been successfully created.")

def process_yoda_data(lines):
    # Prepare containers
    edges = []
    values = []
    errors_dn = []
    errors_up = []
    parsing_values = False
    table_name = "default"

    for line in lines:
        line = line.strip()

        # Table name
        if line.startswith("BEGIN YODA_BINNEDESTIMATE"):
            table_name = line.split()[-1].replace("/", "_")

        # Bin max edges
        if line.startswith("Edges(A1):"):
            match = re.search(r'\[([^\]]+)\]', line)
            if match:
                edge_parts = match.group(1).split(",")
                edges = [float(e.strip().strip('"')) for e in edge_parts]

        # Start of values + error reading
        if line.startswith("# value"):
            parsing_values = True
            continue

        # Read value + errDn + errUp from same line
        if parsing_values and line and not line.startswith("#") and "END" not in line:
            parts = line.split()
        
            # value
            val = parts[0]
            if val.lower() in ["---", "nan"]:
                continue

            values.append(float(val))

            # errDn
            if len(parts) >= 2:
                errdn = parts[1]
                errors_dn.append(0.0 if errdn in ["---", "nan"] else abs(float(errdn)))
            else:
                errors_dn.append(0.0)

            # errUp
            if len(parts) >= 3:
                errup = parts[2]
                errors_up.append(0.0 if errup in ["---", "nan"] else abs(float(errup)))
            else:
                errors_up.append(0.0)

    # Binning
    max_vals = edges[:len(values)]
    min_vals = [0.0] + max_vals[:-1]

    # Write ADL file
    adl_filename = "output_from_yoda.adl"

    with open(adl_filename, "w") as adl_file:
        adl_file.write(f"table {table_name}_table\n")
        adl_file.write(f"tabletype {table_name}\n")
        adl_file.write("nvars 1\n")
        adl_file.write("errors true\n")
        adl_file.write("# {0:<12} {1:<12} {2:<12} {3:<12} {4:<12}\n".format("eff", "err-", "err+", "Min", "Max"))

        n = min(len(values), len(errors_dn), len(errors_up), len(min_vals), len(max_vals))
        for i in range(n):
            adl_file.write("{0:<12.6e} {1:<12.4f} {2:<12.4f} {3:<12.3f} {4:<12.3f}\n".format(values[i], errors_dn[i], errors_up[i], min_vals[i], max_vals[i]))

    
    print(f"ADL file '{adl_filename}' has been successfully created from YODA.")

# Main part

mode = input("Do you want to provide a local file or a URL? (Enter 'local' or 'url'): ").strip().lower()

if mode == "local":
    path = input("Enter the full path of the local file: ").strip()
    if not os.path.exists(path):
        raise FileNotFoundError("File not found.")
    if path.endswith(".yaml"):
        with open(path, "r") as file:
            yaml_content = file.read()
        process_yaml_data(yaml_content)
    elif path.endswith(".yoda"):
        with open(path, "r") as f:
            lines = f.readlines()
        process_yoda_data(lines)
    else:
        raise ValueError("File must end with .yaml or .yoda")
elif mode == "url":
    file_type = input("Which file type? (yaml or yoda): ").strip().lower()
    inspire_id = input("Enter Inspire ID (e.g. 1234567): ").strip()
    table_number = input("Enter Table number (e.g. 2 for Table2): ").strip()
    version_number = input("Enter version number (0 for latest): ").strip()
    if version_number == "0":
        url = f"https://www.hepdata.net/record/ins{inspire_id}?format={file_type}&table=Table{table_number}"
    else:
        url = f"https://www.hepdata.net/record/ins{inspire_id}?format={file_type}&table=Table{table_number}&version={version_number}"
    response = requests.get(url)
    if response.status_code != 200:
        raise Exception(f"Failed to fetch data from URL. Status code: {response.status_code}")
    if "yaml" in url:
        process_yaml_data(response.text)
    elif "yoda" in url:
        process_yoda_data(response.text.split("\n"))
    else:
        raise ValueError("URL must contain .yaml or .yoda file")
else:
    raise ValueError("Invalid choice. Enter 'local' or 'url'")
