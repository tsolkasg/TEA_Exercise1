import glob
import codecs

input_data_path = 'D:/Data_Original/Original/'
output_data_path = 'D:/Data/'

data_type = "train"  # which files to read/write
content = ""

# Read all files in input_data_path
for file in glob.glob(input_data_path+data_type+"/"+"*.txt"):
    with codecs.open(file, encoding='utf-8') as f:
        for line in f.readlines():
            # Remove empty lines and xml tags
            if (line.isspace() | line.strip().startswith("<")):
                continue
            content += line.lower()

# Write the result to output_data_path
text_file = open(output_data_path+data_type+".txt", "wb")
text_file.write(content.encode('utf8'))
text_file.close()



