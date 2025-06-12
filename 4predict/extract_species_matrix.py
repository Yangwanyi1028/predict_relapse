import pandas as pd
import csv
import sys

def extract_species_rows(input_file, output_file):
    species_rows = []
    
    # Read the data with pandas
    try:
        data = pd.read_csv(input_file)
        
        # Extract rows with species information and simplify taxonomy to species name only
        for index, row in data.iterrows():
            taxonomy = row.iloc[0]  # Assuming taxonomy is in the first column
            if isinstance(taxonomy, str) and "s__" in taxonomy:
                # Extract just the species name after "s__"
                parts = taxonomy.split("|")
                for part in parts:
                    if part.startswith("s__"):
                        species_name = part[3:]  # Remove "s__" prefix
                        row_copy = row.copy()
                        row_copy.iloc[0] = species_name
                        species_rows.append(row_copy)
                        break
        
        # Create a new dataframe and check for duplicates
        if species_rows:
            species_df = pd.DataFrame(species_rows)
            
            # Check for duplicate species and merge if found
            if species_df.iloc[:, 0].duplicated().any():
                print("发现重复物种。正在合并重复行...")
                # Group by species name (first column) and aggregate other columns
                # For numerical columns, take the mean; for non-numerical columns, take the first value
                agg_dict = {}
                for col in species_df.columns[1:]:
                    if pd.api.types.is_numeric_dtype(species_df[col]):
                        agg_dict[col] = 'mean'
                    else:
                        agg_dict[col] = 'first'
                
                # Group by species name and aggregate
                original_count = len(species_df)
                species_df = species_df.groupby(species_df.columns[0]).agg(agg_dict).reset_index()
                merged_count = original_count - len(species_df)
                print(f"已合并 {merged_count} 个重复行")
            
            species_df.to_csv(output_file, index=False)
            print(f"提取了 {len(species_df)} 个物种级行到 {output_file}")
        else:
            print("未找到物种级行。")
            
    except Exception as e:
        print(f"使用pandas时出错: {e}")
        # If pandas approach fails, try manual CSV parsing
        with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
            csv_reader = csv.reader(infile)
            csv_writer = csv.writer(outfile)
            
            # Get header and write it to output
            header = next(csv_reader)
            csv_writer.writerow(header)
            
            # Dictionary to store merged rows
            merged_rows = {}
            
            for row in csv_reader:
                if row and "s__" in row[0]:
                    # Extract just the species name
                    taxonomy = row[0]
                    parts = taxonomy.split("|")
                    for part in parts:
                        if part.startswith("s__"):
                            species_name = part[3:]  # Remove "s__" prefix
                            modified_row = row.copy()
                            modified_row[0] = species_name
                            
                            # Store the row with species name as the key
                            if species_name in merged_rows:
                                print(f"发现物种 {species_name} 的重复行。保留第一次出现的行。")
                            else:
                                merged_rows[species_name] = modified_row
                            break
            
            # Write the merged rows to the output file
            for row in merged_rows.values():
                csv_writer.writerow(row)
            
            print(f"提取了 {len(merged_rows)} 个物种级行到 {output_file}")

if __name__ == "__main__":
    input_file = "/Users/yangkeyi/Downloads/predict_relapse/4predict/lefse_diff_abundance_matrix.csv"
    output_file = "lefse_diff_abundance_matrix_sp_only.csv"
    
    if len(sys.argv) > 2:
        input_file = sys.argv[1]
        output_file = sys.argv[2]
        
    extract_species_rows(input_file, output_file)


import pandas as pd
import csv
import sys

def extract_species_rows(input_file, output_file):
    species_rows = []
    
    # Read the data with pandas
    try:
        data = pd.read_csv(input_file)
        
        # Extract rows with species information
        for index, row in data.iterrows():
            taxonomy = row.iloc[0]  # Assuming taxonomy is in the first column
            if isinstance(taxonomy, str) and "s__" in taxonomy:
                species_rows.append(row)
        
        # Create a new dataframe and save to CSV
        if species_rows:
            species_df = pd.DataFrame(species_rows)
            species_df.to_csv(output_file, index=False)
            print(f"Extracted {len(species_rows)} species-level rows to {output_file}")
        else:
            print("No species-level rows found.")
            
    except Exception as e:
        # If pandas approach fails, try manual CSV parsing
        with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
            counter = 0
            for line in infile:
                if "s__" in line:
                    outfile.write(line)
                    counter += 1
            print(f"Extracted {counter} species-level rows to {output_file}")

if __name__ == "__main__":
    input_file = "/Users/yangkeyi/Downloads/predict_relapse/4predict/lefse_diff_abundance_matrix.csv"
    output_file = "/Users/yangkeyi/Downloads/predict_relapse/4predict/lefse_diff_abundance_matrix_sp_only1.csv"
    
    if len(sys.argv) > 2:
        input_file = sys.argv[1]
        output_file = sys.argv[2]
        
    extract_species_rows(input_file, output_file)