import csv

from matrx.logger.logger import GridWorldLogger
from matrx.grid_world import GridWorld

class TrustEvaluationLogger(GridWorldLogger):
    '''
    Logger for saving the actions of all agents during each tick of the task.
    '''
    def __init__(self, save_path="", file_name_prefix="", file_extension=".csv", delimiter=";"):
        super().__init__(save_path=save_path, file_name=file_name_prefix, file_extension=file_extension, delimiter=delimiter, log_strategy=1)

    def log(self, grid_world, agent_data):
        # Create a dictionary with the log data
        log_data = {}

        # Read the current trust beliefs
        with open('beliefs/currentTrustBelief.csv') as csvfile:
            reader = csv.reader(csvfile, delimiter=';', quotechar="'")
            trustfile_header = []
            trustBeliefs = {}
            for row in reader:
                if trustfile_header == []:
                    trustfile_header = row
                    continue
                # Retrieve trust values
                if row:
                    name = row[0]
                    task = row[1]
                    competence = float(row[2])
                    willingness = float(row[3])
                    if name not in trustBeliefs:
                        trustBeliefs[name] = {}

                    trustBeliefs[name][task] = {'competence': competence, 'willingness': willingness}

        # We will log trust beliefs per task
        log_data['name'] = name
        for task in trustBeliefs[name].keys():
            log_data[task + '_competence'] = trustBeliefs[name][task]['competence']
            log_data[task + '_willingness'] = trustBeliefs[name][task]['willingness']


        return log_data