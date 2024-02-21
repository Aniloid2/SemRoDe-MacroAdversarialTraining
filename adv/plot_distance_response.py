import matplotlib.pyplot as plt
import os

# Initialize empty lists to store OT and MMD values for two files
OT1 = []
OT2 = []
MMD1 = []
MMD2 = []

# Define the file locations
file1 = "/home/brianformento/security_project/defense/attention_align/ot_alignment_attention/adv/LOCAL_Test/GLUE/MR/ROBERTA/ARR_OCT_2023/OT_GL_DMR_AKTTextFooler_MR0_3_ATETextFooler_B1_0_AT0_0_OT0_0_MMD1_0_CRL0_0_AUGNone_P0_0_PF0_0_FLB0_0_OAT_None_DR0_1_E7_LR2e-05_SS0_5_0_5_CS0_5_0_5_NC50_50_NWS0_FT/L.txt"
file2 = "/home/brianformento/security_project/defense/attention_align/ot_alignment_attention/adv/LOCAL_Test/GLUE/MR/BERT/Pooled_MR/OT_GL_DMR_AKTTextFooler_MR0_3_ATETextFooler_B1_0_AT0_0_OT0_0_MMD1_0_CRL0_0_AUGNone_P0_0_PF0_0_FLB0_0_DR0_1_E7_LR2e-05_SS0_5_0_5_CS0_5_0_5_NC50_50_NWS0_FT/L.txt"

# file1 = "/home/brianformento/security_project/defense/attention_align/ot_alignment_attention/adv/LOCAL_Test/GLUE/AGNEWS/ROBERTA/ARR_OCT_2023/OT_GL_DAGNEWS_AKTTextFooler_MR0_3_ATETextFooler_B1_0_AT0_0_OT0_0_MMD1_0_CRL0_0_AUGNone_P0_0_PF0_0_FLB0_0_OAT_None_DR0_1_E3_LR2e-05_SS0_5_0_5_CS0_5_0_5_NC50_50_NWS0_FT/L.txt"
# file2 = "/home/brianformento/security_project/defense/attention_align/ot_alignment_attention/adv/LOCAL_Test/GLUE/AGNEWS/BERT/Pooled_AGNEWS/OT_GL_DAGNEWS_AKTTextFooler_MR0_3_ATETextFooler_B1_0_AT0_0_OT0_0_MMD1_0_CRL0_0_AUGNone_P0_0_PF0_0_FLB0_0_DR0_1_E3_LR2e-05_SS0_5_0_5_CS0_5_0_5_NC50_50_NWS0_FT/L.txt"

# file1 = "/home/brianformento/security_project/defense/attention_align/ot_alignment_attention/adv/LOCAL_Test/GLUE/SST2/ROBERTA/ARR_OCT_2023_Rigth_env_new_roberta_classification_head/OT_GL_DSST2_AKTTextFooler_MR0_3_ATETextFooler_B1_0_AT0_0_OT0_0_MMD1_0_CRL0_0_AUGNone_P0_0_PF0_0_FLB0_0_OAT_None_DR0_1_E3_LR2e-05_SS0_5_0_5_CS0_5_0_5_NC50_50_NWS0_FT/L.txt"
# file2 = "/home/brianformento/security_project/defense/attention_align/ot_alignment_attention/adv/LOCAL_Test/GLUE/SST2/BERT/ARR_OCT_2023/OT_GL_DSST2_AKTTextFooler_MR0_3_ATETextFooler_B1_0_AT0_0_OT0_0_MMD1_0_CRL0_0_AUGNone_P0_0_PF0_0_FLB0_0_OAT_None_DR0_1_E3_LR2e-05_SS0_5_0_5_CS0_5_0_5_NC50_50_NWS0_FT/L.txt"

# Loop over each file
for i, file_location in enumerate([file1, file2]):
    # Open the file located at the path
    with open(file_location, 'r') as file:
        # Read each line in the file
        for line in file:
            # Split the line into a list of values
            values = line.split(',')

            # Get the OT and MMD value, strip unnecessary spaces and convert it to a float type
            OT_value = float(values[3].strip().split(':')[1])
            MMD_value = float(values[5].strip().split(':')[1])

            # Append the values to the correct lists depending on the file
            if i == 0:
                OT1.append(OT_value)
                MMD1.append(MMD_value)
            elif i == 1:
                OT2.append(OT_value)
                MMD2.append(MMD_value)

# Extract dataset name from file location (6th position in the list when split by "/")
dataset_name = file1.split("/")[10]

# Create a list for iterations
iterations = list(range(1, len(OT1) + 1))

# Plot OT against iterations for the two files
plt.figure()
plt.plot(iterations, OT1, label="Roberta", color='blue')
plt.plot(iterations, OT2, label="BERT", color='red')
plt.xlabel('Iteration Number')
plt.ylabel('Wasserstein Distance')
# plt.title('Wasserstein Distance values across iterations for {} dataset'.format(dataset_name))
plt.legend()

# Generate a unique filename for OT plot and save it in both png and svg format
new_directory = "/home/brianformento/security_project/defense/attention_align/ot_alignment_attention/adv/Distance_Response/"
os.makedirs(new_directory, exist_ok=True)
filename = "OT_{}_comparison.png".format(dataset_name)
file_path = os.path.join(new_directory, filename)
plt.savefig(file_path)
filename_svg = "SVG_OT_{}_comparison.eps".format(dataset_name)
file_path_svg = os.path.join(new_directory, filename_svg)
plt.savefig(file_path_svg, format='eps')

# Plot MMD against iterations
plt.figure()
plt.plot(iterations, MMD1, label="Roberta", color='blue')
plt.plot(iterations, MMD2, label="BERT", color='red')
plt.xlabel('Iteration Number')
plt.ylabel('MMD Value')
# plt.title('MMD values across iterations for {} dataset'.format(dataset_name))
plt.legend()

# Generate a unique filename for MMD plot and save it in both png and svg format
filename = "MMD_{}_comparison.png".format(dataset_name)
file_path = os.path.join(new_directory, filename)
plt.savefig(file_path)
filename_svg = "SVG_MMD_{}_comparison.eps".format(dataset_name)
file_path_svg = os.path.join(new_directory, filename_svg)
plt.savefig(file_path_svg, format='eps')