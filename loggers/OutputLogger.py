import os, requests
import sys
import csv
import glob
import pathlib
import matplotlib.pyplot as plt

def output_logger(fld):
    recent_dir = max(glob.glob(os.path.join(fld, '*/')), key=os.path.getmtime)
    recent_dir = max(glob.glob(os.path.join(recent_dir, '*/')), key=os.path.getmtime)
    action_files = glob.glob(os.path.join(recent_dir, 'world_1/action*'))
    if action_files:
        action_file = action_files[0]
    else:
        print(f"No action files found in {os.path.join(recent_dir, 'world_1')}")
        return
    action_header = []
    action_contents=[]
    trustfile_header = []
    trustfile_contents = []
    # Calculate the unique human and agent actions
    unique_agent_actions = []
    unique_human_actions = []
    with open(action_file) as csvfile:
        reader = csv.reader(csvfile, delimiter=';', quotechar="'")
        for row in reader:
            if action_header==[]:
                action_header=row
                continue
            if row[2:4] not in unique_agent_actions and row[2]!="":
                unique_agent_actions.append(row[2:4])
            if row[4:6] not in unique_human_actions and row[4]!="":
                unique_human_actions.append(row[4:6])
            if row[4] == 'RemoveObjectTogether' or row[4] == 'CarryObjectTogether' or row[4] == 'DropObjectTogether':
                if row[4:6] not in unique_agent_actions:
                    unique_agent_actions.append(row[4:6])
            res = {action_header[i]: row[i] for i in range(len(action_header))}
            action_contents.append(res)

    with open(fld+'/beliefs/currentTrustBelief.csv') as csvfile:
        reader = csv.reader(csvfile, delimiter=';', quotechar="'")
        for row in reader:
            if trustfile_header==[]:
                trustfile_header=row
                continue
            if row:
                res = {trustfile_header[i] : row[i] for i in range(len(trustfile_header))}
                trustfile_contents.append(res)
    # Retrieve the stored trust belief values
    name = trustfile_contents[-1]['name']
    task = trustfile_contents[-1]['task']
    competence = trustfile_contents[-1]['competence']
    willingness = trustfile_contents[-1]['willingness']
    # Retrieve the number of ticks to finish the task, score, and completeness
    no_ticks = action_contents[-1]['tick_nr']
    score = action_contents[-1]['score']
    completeness = action_contents[-1]['completeness']
    # Save the output as a csv file
    print("Saving output...")
    with open(os.path.join(recent_dir,'world_1/output.csv'),mode='w') as csv_file:
        csv_writer = csv.writer(csv_file, delimiter=';', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        csv_writer.writerow(['completeness','score','no_ticks','agent_actions','human_actions'])
        csv_writer.writerow([completeness,score,no_ticks,len(unique_agent_actions),len(unique_human_actions)])
    with open(fld + '/beliefs/allTrustBeliefs.csv', mode='a+') as csv_file:
        csv_writer = csv.writer(csv_file, delimiter=';', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        csv_writer.writerow([name,task,competence,willingness])

    evaluation_plots(recent_dir)


def evaluation_plots(recent_dir):
    files = glob.glob(os.path.join(recent_dir, 'world_1/evaluation_*'))
    if len(files) < 0:
        return

    evaluation_filepath = files[0]
    ticks = []
    trust_beliefs = {}
    with open(evaluation_filepath) as csv_file:
        reader = csv.reader(csv_file, delimiter=';', quotechar="'")
        header = next(reader)
        trust_beliefs[header[1]] = []
        trust_beliefs[header[2]] = []
        trust_beliefs[header[3]] = []
        trust_beliefs[header[4]] = []

        # skip over the first row
        next(reader)

        for i, row in enumerate(reader):
            if i % 100 == 0:
                ticks.append(i)
                trust_beliefs[header[1]].append(float(row[1]))
                trust_beliefs[header[2]].append(float(row[2]))
                trust_beliefs[header[3]].append(float(row[3]))
                trust_beliefs[header[4]].append(float(row[4]))

    plt.figure(figsize=(10, 10))

    # Create a plot for the search task
    plt.subplot(2, 1, 1)  # (rows, columns, index) → First subplot
    plt.plot(ticks, trust_beliefs[header[1]], marker='o', linestyle='-', label=header[1])
    plt.plot(ticks, trust_beliefs[header[2]], marker='o', linestyle='-', label=header[2])
    plt.xlabel("Ticks")
    plt.ylabel("Trust")
    plt.title("Search Trust Values Throughout the Game")
    plt.legend()
    plt.grid(True)

    # Second plot: header[3] and header[4]
    plt.subplot(2, 1, 2)  # Second subplot below the first one
    plt.plot(ticks, trust_beliefs[header[3]], marker='o', linestyle='-', label=header[3])
    plt.plot(ticks, trust_beliefs[header[4]], marker='o', linestyle='-', label=header[4])
    plt.xlabel("Ticks")
    plt.ylabel("Trust")
    plt.title("Rescue Trust Values Throughout the Game")
    plt.legend()
    plt.grid(True)

    # Save and show
    plt.tight_layout()  # Adjust layout to prevent overlap
    plt.savefig(os.path.join(recent_dir, 'world_1/evaluation_plot.png'))
    plt.show()
