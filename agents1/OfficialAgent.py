import csv
import enum
import re

from matrx import utils
from matrx.agents.agent_utils.navigator import Navigator
from matrx.agents.agent_utils.state_tracker import StateTracker
from matrx.messages.message import Message

from actions1.CustomActions import *
from actions1.CustomActions import CarryObject, Drop
from brains1.ArtificialBrain import ArtificialBrain


class Phase(enum.Enum):
    INTRO = 1,
    FIND_NEXT_GOAL = 2,
    PICK_UNSEARCHED_ROOM = 3,
    PLAN_PATH_TO_ROOM = 4,
    FOLLOW_PATH_TO_ROOM = 5,
    PLAN_ROOM_SEARCH_PATH = 6,
    FOLLOW_ROOM_SEARCH_PATH = 7,
    PLAN_PATH_TO_VICTIM = 8,
    FOLLOW_PATH_TO_VICTIM = 9,
    TAKE_VICTIM = 10,
    PLAN_PATH_TO_DROPPOINT = 11,
    FOLLOW_PATH_TO_DROPPOINT = 12,
    DROP_VICTIM = 13,
    WAIT_FOR_HUMAN = 14,
    WAIT_AT_ZONE = 15,
    FIX_ORDER_GRAB = 16,
    FIX_ORDER_DROP = 17,
    REMOVE_OBSTACLE_IF_NEEDED = 18,
    ENTER_ROOM = 19


class BaselineAgent(ArtificialBrain):
    def __init__(self, slowdown, condition, name, folder):
        super().__init__(slowdown, condition, name, folder)
        # Initialization of some relevant variables
        self._current_tick_received_messages = []
        self._send_message_ticks = []
        self._state = None
        self._last_received_message = None
        self._tick_distance_goal = {}
        self._tick = None
        self._slowdown = slowdown
        self._condition = condition
        self._human_name = name
        self._folder = folder
        self._phase = Phase.INTRO
        self._room_vics = []
        self._searched_rooms = set()
        self._found_victims = []
        self._collected_victims = []
        self._found_victim_logs = {}
        self._send_messages = []
        self._current_door = None
        self._team_members = []
        self._carrying_together = False
        self._remove = False
        self._goal_vic = None
        self._goal_loc = None
        self._human_loc = None
        self._distance_human = None
        self._distance_drop = None
        self._agent_loc = None
        self._todo = []
        self._answered = False
        self._to_search = []
        self._carrying = False
        self._waiting = False
        self._rescue = None
        self._recent_vic = None
        self._received_messages = []
        self._moving = False
        self._default_trust_value = 0
        self.robot_found = False
        # avoid loops and unnecessary checks in phase.take_victim
        self._take_victim_repeat = False
        self._last_length_received_messages = 0
        self._last_length_send_messages = 0



        self._lookup_table = {
            "Mild": {
                "Rescue together": {
                    ("close", "close"): {"competence": 0.1, "willingness": 0.1},
                    ("close", "far"): {"competence": 0.2, "willingness": 0.1},
                    ("far", "close"): {"competence": -0.05, "willingness": 0.15},
                    ("far", "far"): {"competence": 0.1, "willingness": 0.15},
                },
                "Rescue alone": {
                    ("close", "close"): {"competence": 0.1, "willingness": -0.15},
                    ("close", "far"): {"competence": -0.1, "willingness": -0.15},
                    ("far", "close"): {"competence": 0.2, "willingness": -0.1},
                    ("far", "far"): {"competence": 0.1, "willingness": -0.1},
                },

                "Continue": {
                    ("close", "close"): {"competence": -0.1, "willingness": -0.1},
                    ("close", "far"): {"competence": -0.15, "willingness": -0.1},
                    ("far", "close"): {"competence": -0.1, "willingness": -0.15},
                    ("far", "far"): {"competence": 0.05, "willingness": -0.15},
                },
            },
            "Critical": {
                "Rescue": {  # For "critical" cases
                    ("close", "close"): {"competence": 0.15, "willingness": 0.1},
                    ("close", "far"): {"competence": 0.2, "willingness": 0.1},
                    ("far", "close"): {"competence": 0.1, "willingness": 0.15},
                    ("far", "far"): {"competence": 0.15, "willingness": 0.15},
                },
                "Continue": {  # For "Continue" in critical cases
                    ("close", "close"): {"competence": -0.15, "willingness": -0.15},
                    ("close", "far"): {"competence": -0.2, "willingness": -0.15},
                    ("far", "close"): {"competence": -0.1, "willingness": -0.1},
                    ("far", "far"): {"competence": -0.1, "willingness": -0.1},
                },
            },
        }

    def initialize(self):
        # Initialization of the state tracker and navigation algorithm
        self._state_tracker = StateTracker(agent_id=self.agent_id)
        self._navigator = Navigator(agent_id=self.agent_id, action_set=self.action_set,
                                    algorithm=Navigator.A_STAR_ALGORITHM)

    def filter_observations(self, state):
        # Filtering of the world state before deciding on an action 
        return state

    def decide_on_actions(self, state):
        # Identify team members
        agent_name = state[self.agent_id]['obj_id']
        for member in state['World']['team_members']:
            if member != agent_name and member not in self._team_members:
                self._team_members.append(member)
        # Create a list of received messages from the human team member
        for mssg in self.received_messages:
            for member in self._team_members:
                if mssg.from_id == member and mssg.content not in self._received_messages:
                    self._received_messages.append(mssg.content)

        # Process messages from team members
        self._process_messages(state, self._team_members, self._condition)
        # Initialize and update trust beliefs for team members
        trustBeliefs = self._loadBelief(self._team_members, self._folder + '/beliefs/allTrustBeliefs.csv')
        self._trustBelief(self._team_members, trustBeliefs, self._folder, self._received_messages)

        # Check whether human is close in distance
        if state[{'is_human_agent': True}]:
            self._distance_human = 'close'
        if not state[{'is_human_agent': True}]:
            # Define distance between human and agent based on last known area locations
            if self._agent_loc in [1, 2, 3, 4, 5, 6, 7] and self._human_loc in [8, 9, 10, 11, 12, 13, 14]:
                self._distance_human = 'far'
            if self._agent_loc in [1, 2, 3, 4, 5, 6, 7] and self._human_loc in [1, 2, 3, 4, 5, 6, 7]:
                self._distance_human = 'close'
            if self._agent_loc in [8, 9, 10, 11, 12, 13, 14] and self._human_loc in [1, 2, 3, 4, 5, 6, 7]:
                self._distance_human = 'far'
            if self._agent_loc in [8, 9, 10, 11, 12, 13, 14] and self._human_loc in [8, 9, 10, 11, 12, 13, 14]:
                self._distance_human = 'close'

        # Define distance to drop zone based on last known area location
        if self._agent_loc in [1, 2, 5, 6, 8, 9, 11, 12]:
            self._distance_drop = 'far'
        if self._agent_loc in [3, 4, 7, 10, 13, 14]:
            self._distance_drop = 'close'

        # Check whether victims are currently being carried together by human and agent 
        for info in state.values():
            if 'is_human_agent' in info and self._human_name in info['name'] and len(
                    info['is_carrying']) > 0 and 'critical' in info['is_carrying'][0]['obj_id'] or \
                    'is_human_agent' in info and self._human_name in info['name'] and len(
                info['is_carrying']) > 0 and 'mild' in info['is_carrying'][0][
                'obj_id'] and self._rescue == 'together' and not self._moving:
                # If victim is being carried, add to collected victims memory
                if info['is_carrying'][0]['img_name'][8:-4] not in self._collected_victims:
                    self._collected_victims.append(info['is_carrying'][0]['img_name'][8:-4])
                self._carrying_together = True
            if 'is_human_agent' in info and self._human_name in info['name'] and len(info['is_carrying']) == 0:
                self._carrying_together = False
        # If carrying a victim together, let agent be idle (because joint actions are essentially carried out by the human)
        if self._carrying_together == True:
            return None, {}

        # Send the hidden score message for displaying and logging the score during the task, DO NOT REMOVE THIS
        self._send_message('Our score is ' + str(state['rescuebot']['score']) + '.', 'RescueBot')

        # Ongoing loop until the task is terminated, using different phases for defining the agent's behavior
        while True:
            if Phase.INTRO == self._phase:
                # Send introduction message
                self._send_message('Hello! My name is RescueBot. Together we will collaborate and try to search and rescue the 8 victims on our right as quickly as possible. \
                Each critical victim (critically injured girl/critically injured elderly woman/critically injured man/critically injured dog) adds 6 points to our score, \
                each mild victim (mildly injured boy/mildly injured elderly man/mildly injured woman/mildly injured cat) 3 points. \
                If you are ready to begin our mission, you can simply start moving.', 'RescueBot')
                # Wait until the human starts moving before going to the next phase, otherwise remain idle
                if not state[{'is_human_agent': True}]:
                    self._phase = Phase.FIND_NEXT_GOAL
                else:
                    return None, {}

            if Phase.FIND_NEXT_GOAL == self._phase:
                # Definition of some relevant variables
                self._answered = False
                self._goal_vic = None
                self._goal_loc = None
                self._rescue = None
                self._moving = True
                remaining_zones = []
                remaining_vics = []
                remaining = {}
                # Identification of the location of the drop zones
                zones = self._get_drop_zones(state)
                # Identification of which victims still need to be rescued and on which location they should be dropped
                for info in zones:
                    if str(info['img_name'])[8:-4] not in self._collected_victims:
                        remaining_zones.append(info)
                        remaining_vics.append(str(info['img_name'])[8:-4])
                        remaining[str(info['img_name'])[8:-4]] = info['location']
                if remaining_zones:
                    self._remainingZones = remaining_zones
                    self._remaining = remaining
                    # if self._collected_victims == 8:
                    #     self._send_message('There are still free zones, but all victims should have been collected based on your messages', 'RescueBot')
                # Remain idle if there are no victims left to rescue

                if not remaining_zones:
                    return None, {}


                # Check which victims can be rescued next because human or agent already found them
                for vic in remaining_vics:
                    # Define a previously found victim as target victim because all areas have been searched
                    if vic in self._found_victims and vic in self._todo and len(self._searched_rooms) == 0:
                        self._goal_vic = vic
                        self._goal_loc = remaining[vic]
                        # Move to target victim
                        self._rescue = 'together'
                        self._send_message('Moving to ' + self._found_victim_logs[vic][
                            'room'] + ' to pick up ' + self._goal_vic + '. Please come there as well to help me carry ' + self._goal_vic + ' to the drop zone.',
                                          'RescueBot')
                        # Plan path to victim because the exact location is known (i.e., the agent found this victim)
                        if 'location' in self._found_victim_logs[vic].keys():
                            self.robot_found = True
                            self._phase = Phase.PLAN_PATH_TO_VICTIM
                            return Idle.__name__, {'duration_in_ticks': 25}
                        # Plan path to area because the exact victim location is not known, only the area (i.e., human found this  victim)
                        if 'location' not in self._found_victim_logs[vic].keys():
                            self._phase = Phase.PLAN_PATH_TO_ROOM
                            return Idle.__name__, {'duration_in_ticks': 25}
                    # Define a previously found victim as target victim
                    if vic in self._found_victims and vic not in self._todo:
                        self._goal_vic = vic
                        self._goal_loc = remaining[vic]
                        # Rescue together when victim is critical or when the human is weak and the victim is mildly injured
                        if 'critical' in vic or 'mild' in vic and self._condition == 'weak':
                            self._rescue = 'together'
                        # Rescue alone if the victim is mildly injured and the human not weak
                        if 'mild' in vic and self._condition != 'weak':
                            self._rescue = 'alone'
                        # Plan path to victim because the exact location is known (i.e., the agent found this victim)
                        if 'location' in self._found_victim_logs[vic].keys():
                            self.robot_found = True
                            self._phase = Phase.PLAN_PATH_TO_VICTIM
                            return Idle.__name__, {'duration_in_ticks': 25}
                        # Plan path to area because the exact victim location is not known, only the area (i.e., human found this  victim)
                        if 'location' not in self._found_victim_logs[vic].keys():
                            self._phase = Phase.PLAN_PATH_TO_ROOM
                            return Idle.__name__, {'duration_in_ticks': 25}
                    # If there are no target victims found, visit an unsearched area to search for victims
                    if vic not in self._found_victims or vic in self._found_victims and vic in self._todo and len(
                            self._searched_rooms) > 0:
                        self._phase = Phase.PICK_UNSEARCHED_ROOM

            if Phase.PICK_UNSEARCHED_ROOM == self._phase:
                agent_location = state[self.agent_id]['location']
                # Identify which areas are not explored yet
                unsearched_rooms = [room['room_name'] for room in state.values()
                                   if 'class_inheritance' in room
                                   and 'Door' in room['class_inheritance']
                                   and room['room_name'] not in self._searched_rooms
                                   and room['room_name'] not in self._to_search]
                # If all areas have been searched but the task is not finished, start searching areas again
                if self._remainingZones and len(unsearched_rooms) == 0:
                    self._to_search = []
                    self._searched_rooms = set()
                    self._last_length_send_messages = len(self._send_messages)
                    self._send_messages = []
                    self._last_length_received_messages = len(self.received_messages)
                    self.received_messages = []
                    self.received_messages_content = []
                    self._send_message('Going to re-search all areas.', 'RescueBot')
                    self._phase = Phase.FIND_NEXT_GOAL
                # If there are still areas to search, define which one to search next
                else:
                    # Identify the closest door when the agent did not search any areas yet
                    if self._current_door == None:
                        # Find all area entrance locations
                        self._door = state.get_room_doors(self._getClosestRoom(state, unsearched_rooms, agent_location))[
                            0]
                        self._doormat = \
                            state.get_room(self._getClosestRoom(state, unsearched_rooms, agent_location))[-1]['doormat']
                        # Workaround for one area because of some bug
                        if self._door['room_name'] == 'area 1':
                            self._doormat = (3, 5)
                        # Plan path to area
                        self._phase = Phase.PLAN_PATH_TO_ROOM
                    # Identify the closest door when the agent just searched another area
                    if self._current_door != None:
                        self._door = \
                            state.get_room_doors(self._getClosestRoom(state, unsearched_rooms, self._current_door))[0]
                        self._doormat = \
                            state.get_room(self._getClosestRoom(state, unsearched_rooms, self._current_door))[-1][
                                'doormat']
                        if self._door['room_name'] == 'area 1':
                            self._doormat = (3, 5)
                        self._phase = Phase.PLAN_PATH_TO_ROOM

            if Phase.PLAN_PATH_TO_ROOM == self._phase:

                # Reset the navigator for a new path planning
                self._navigator.reset_full()

                # Check if there is a goal victim, and it has been found, but its location is not known
                if self._goal_vic \
                        and self._goal_vic in self._found_victims \
                        and 'location' not in self._found_victim_logs[self._goal_vic].keys():
                    # Retrieve the victim's room location and related information
                    victim_location = self._found_victim_logs[self._goal_vic]['room']
                    self._door = state.get_room_doors(victim_location)[0]
                    self._doormat = state.get_room(victim_location)[-1]['doormat']

                    # Handle special case for 'area 1'
                    if self._door['room_name'] == 'area 1':
                        self._doormat = (3, 5)

                    # Set the door location based on the doormat
                    doorLoc = self._doormat

                # If the goal victim's location is known, plan the route to the identified area
                else:
                    if self._door['room_name'] == 'area 1':
                        self._doormat = (3, 5)
                    doorLoc = self._doormat

                # Add the door location as a waypoint for navigation
                self._navigator.add_waypoints([doorLoc])
                # Follow the route to the next area to search
                self._phase = Phase.FOLLOW_PATH_TO_ROOM

            if Phase.FOLLOW_PATH_TO_ROOM == self._phase:
                # Check if the previously identified target victim was rescued by the human
                if self._goal_vic and self._goal_vic in self._collected_victims:
                    # Reset current door and switch to finding the next goal
                    self._current_door = None
                    self._phase = Phase.FIND_NEXT_GOAL

                # Check if the human found the previously identified target victim in a different room
                if self._goal_vic \
                        and self._goal_vic in self._found_victims \
                        and self._door['room_name'] != self._found_victim_logs[self._goal_vic]['room']:
                    self._current_door = None
                    self._phase = Phase.FIND_NEXT_GOAL

                # Check if the human already searched the previously identified area without finding the target victim
                if self._door['room_name'] in self._searched_rooms and self._goal_vic not in self._found_victims:
                    self._current_door = None
                    self._phase = Phase.FIND_NEXT_GOAL

                # Move to the next area to search
                else:
                    # Update the state tracker with the current state
                    self._state_tracker.update(state)

                    # Explain why the agent is moving to the specific area, either:
                    # [-] it contains the current target victim
                    # [-] it is the closest un-searched area
                    if self._goal_vic in self._found_victims \
                            and str(self._door['room_name']) == self._found_victim_logs[self._goal_vic]['room'] \
                            and not self._remove:
                        if self._condition == 'weak':
                            self._send_message('Moving to ' + str(
                                self._door['room_name']) + ' to pick up ' + self._goal_vic + ' together with you.',
                                              'RescueBot')
                        else:
                            self._send_message(
                                'Moving to ' + str(self._door['room_name']) + ' to pick up ' + self._goal_vic + '.',
                                'RescueBot')

                    if self._goal_vic not in self._found_victims and not self._remove or not self._goal_vic and not self._remove:
                        self._send_message(
                            'Moving to ' + str(self._door['room_name']) + ' because it is the closest unsearched area.',
                            'RescueBot')

                    # Set the current door based on the current location
                    self._current_door = self._door['location']

                    # Retrieve move actions to execute
                    action = self._navigator.get_move_action(self._state_tracker)
                    # Check for obstacles blocking the path to the area and handle them if needed
                    if action is not None:
                        # Remove obstacles blocking the path to the area 
                        for info in state.values():
                            if 'class_inheritance' in info and 'ObstacleObject' in info[
                                'class_inheritance'] and 'stone' in info['obj_id'] and info['location'] not in [(9, 4),
                                                                                                                (9, 7),
                                                                                                                (9, 19),
                                                                                                                (21,
                                                                                                                 19)]:
                                self._send_message('Reaching ' + str(self._door['room_name'])
                                                   + ' will take a bit longer because I found stones blocking my path.',
                                                   'RescueBot')
                                return RemoveObject.__name__, {'object_id': info['obj_id']}
                        return action, {}
                    # Identify and remove obstacles if they are blocking the entrance of the area
                    self._phase = Phase.REMOVE_OBSTACLE_IF_NEEDED

            if Phase.REMOVE_OBSTACLE_IF_NEEDED == self._phase:

                objects = []
                agent_location = state[self.agent_id]['location']
                # Identify which obstacle is blocking the entrance
                for info in state.values():
                    if 'class_inheritance' in info and 'ObstacleObject' in info['class_inheritance'] and 'rock' in info[
                        'obj_id']:
                        objects.append(info)
                        # Communicate which obstacle is blocking the entrance
                        if self._answered == False and not self._remove and not self._waiting:
                            self._send_message('Found rock blocking ' + str(self._door['room_name']) + '. Please decide whether to "Remove" or "Continue" searching. \n \n \
                                Important features to consider are: \n safe - victims rescued: ' + str(
                                self._collected_victims) + ' \n explore - areas searched: area ' + str(
                                self._searched_rooms).replace('area ', '') + ' \
                                \n clock - removal time: 5 seconds \n afstand - distance between us: ' + self._distance_human,
                                               'RescueBot')
                            self._waiting = True
                            ## !!!!! ADDED BY US !!!!!

                            # if there is anything to do besides removing the obstacle, enter this if statement
                            # otherwise, behave as before (trust the human unconditionally)
                            if len(self._collected_victims) <6: # if >=6, then likely only the two
                                # rocks are left, so we have to trust the human,
                                # or they cooperated well enough
                                if trustBeliefs[self._human_name]['search']['competence'] < -0.2:
                                    self._send_message('I found a rock blocking ' + str(self._door['room_name']) + '. However, I will ignore it for now and continue searching.','RescueBot')
                                    self._answered = True
                                    self._waiting = False
                                    self._to_search.append(self._door['room_name'])
                                    self._phase = Phase.FIND_NEXT_GOAL
                                # Determine the next area to explore if the human tells the agent not to remove the obstacle
                        if self.received_messages_content and self.received_messages_content[
                            -1] == 'Continue' and not self._remove:
                            self._answered = True
                            self._waiting = False
                            # Add area to the to do list
                            self._to_search.append(self._door['room_name'])
                            self._phase = Phase.FIND_NEXT_GOAL
                        # Wait for the human to help removing the obstacle and remove the obstacle together
                        if self.received_messages_content and self.received_messages_content[
                            -1] == 'Remove' or self._remove:
                            if not self._remove:
                                self._answered = True
                            # Tell the human to come over and be idle untill human arrives
                            if not state[{'is_human_agent': True}]:
                                self._send_message(
                                    'Please come to ' + str(self._door['room_name']) + ' to remove rock.',
                                    'RescueBot')

                                return None, {}
                            # Tell the human to remove the obstacle when he/she arrives
                            if state[{'is_human_agent': True}]:
                                self._send_message('Lets remove rock blocking ' + str(self._door['room_name']) + '!',
                                                   'RescueBot')
                                return None, {}
                        # Remain idle untill the human communicates what to do with the identified obstacle 
                        else:
                            return None, {}

                    if 'class_inheritance' in info and 'ObstacleObject' in info['class_inheritance'] and 'tree' in info[
                        'obj_id']:
                        objects.append(info)
                        # Communicate which obstacle is blocking the entrance
                        if self._answered == False and not self._remove and not self._waiting:
                            self._send_message('Found tree blocking ' + str(self._door['room_name']) + '. Please decide whether to "Remove" or "Continue" searching. \n \n \
                                Important features to consider are: \n safe - victims rescued: ' + str(
                                self._collected_victims) + '\n explore - areas searched: area ' + str(
                                self._searched_rooms).replace('area ', '') + ' \
                                \n clock - removal time: 10 seconds', 'RescueBot')
                            self._waiting = True
                            # TODO: wait ticks for human to respond, if they didnt move on
                        # Determine the next area to explore if the human tells the agent not to remove the obstacle
                        if self.received_messages_content and self.received_messages_content[
                            -1] == 'Continue' and not self._remove:
                            self._answered = True
                            self._waiting = False
                            # Add area to the to do list
                            self._to_search.append(self._door['room_name'])
                            self._phase = Phase.FIND_NEXT_GOAL
                        # Remove the obstacle if the human tells the agent to do so
                        if self.received_messages_content and self.received_messages_content[
                            -1] == 'Remove' or self._remove:
                            if not self._remove:
                                self._answered = True
                                self._waiting = False
                                self._send_message('Removing tree blocking ' + str(self._door['room_name']) + '.',
                                                   'RescueBot')
                            if self._remove:
                                self._send_message('Removing tree blocking ' + str(
                                    self._door['room_name']) + ' because you asked me to.', 'RescueBot')
                            self._phase = Phase.ENTER_ROOM
                            self._remove = False
                            return RemoveObject.__name__, {'object_id': info['obj_id']}
                        # Remain idle untill the human communicates what to do with the identified obstacle
                        else:
                            return None, {}

                    if 'class_inheritance' in info and 'ObstacleObject' in info['class_inheritance'] and 'stone' in \
                            info['obj_id']:
                        objects.append(info)
                        # Communicate which obstacle is blocking the entrance
                        if self._answered == False and not self._remove and not self._waiting:
                            ## !!!!! ADDED BY US !!!!!
                            if trustBeliefs[self._human_name]['search']['competence'] < -0.2:
                                self._send_message('I found stones blocking ' + str(self._door[
                                                                                        'room_name']) + '. However, I will remove it by myself.',
                                                   'RescueBot')
                                self._answered = True
                                self._waiting = False
                                self._phase = Phase.ENTER_ROOM
                                self._remove = False
                                return RemoveObject.__name__, {'object_id': info['obj_id']}

                            self._send_message('Found stones blocking ' + str(self._door['room_name']) + '. Please decide whether to "Remove together", "Remove alone", or "Continue" searching. \n \n \
                                Important features to consider are: \n safe - victims rescued: ' + str(
                                self._collected_victims) + ' \n explore - areas searched: area ' + str(
                                self._searched_rooms).replace('area', '') + ' \
                                \n clock - removal time together: 3 seconds \n afstand - distance between us: ' + self._distance_human + '\n clock - removal time alone: 20 seconds',
                                               'RescueBot')
                            self._waiting = True

                        # Determine the next area to explore if the human tells the agent not to remove the obstacle          
                        if self.received_messages_content and self.received_messages_content[
                            -1] == 'Continue' and not self._remove:
                            self._answered = True
                            self._waiting = False
                            # Add area to the to do list
                            self._to_search.append(self._door['room_name'])
                            self._phase = Phase.FIND_NEXT_GOAL
                        # Remove the obstacle alone if the human decides so
                        if self.received_messages_content and self.received_messages_content[
                            -1] == 'Remove alone' and not self._remove:
                            self._answered = True
                            self._waiting = False
                            self._send_message('Removing stones blocking ' + str(self._door['room_name']) + '.',
                                               'RescueBot')
                            self._phase = Phase.ENTER_ROOM
                            self._remove = False
                            return RemoveObject.__name__, {'object_id': info['obj_id']}
                        # Remove the obstacle together if the human decides so
                        if self.received_messages_content and self.received_messages_content[
                            -1] == 'Remove together' or self._remove:
                            if not self._remove:
                                self._answered = True
                            # Tell the human to come over and be idle untill human arrives
                            if not state[{'is_human_agent': True}]:
                                self._send_message(
                                    'Please come to ' + str(self._door['room_name']) + ' to remove stones together.',
                                    'RescueBot')
                                return None, {}
                            # Tell the human to remove the obstacle when he/she arrives
                            if state[{'is_human_agent': True}]:
                                self._send_message('Lets remove stones blocking ' + str(self._door['room_name']) + '!',
                                                   'RescueBot')
                                return None, {}
                        # Remain idle until the human communicates what to do with the identified obstacle
                        else:
                            return None, {}
                # If no obstacles are blocking the entrance, enter the area
                if len(objects) == 0:
                    self._answered = False
                    self._remove = False
                    self._waiting = False
                    self._phase = Phase.ENTER_ROOM

            if Phase.ENTER_ROOM == self._phase:
                self._answered = False

                # Check if the target victim has been rescued by the human, and switch to finding the next goal
                if self._goal_vic in self._collected_victims:
                    self._current_door = None
                    self._phase = Phase.FIND_NEXT_GOAL

                # Check if the target victim is found in a different area, and start moving there
                if self._goal_vic in self._found_victims \
                        and self._door['room_name'] != self._found_victim_logs[self._goal_vic]['room']:
                    self._current_door = None
                    self._phase = Phase.FIND_NEXT_GOAL

                # Check if area already searched without finding the target victim, and plan to search another area
                if self._door['room_name'] in self._searched_rooms and self._goal_vic not in self._found_victims:
                    self._current_door = None
                    self._phase = Phase.FIND_NEXT_GOAL

                # Enter the area and plan to search it
                else:
                    self._state_tracker.update(state)

                    action = self._navigator.get_move_action(self._state_tracker)
                    # If there is a valid action, return it; otherwise, plan to search the room
                    if action is not None:
                        return action, {}
                    self._phase = Phase.PLAN_ROOM_SEARCH_PATH

            if Phase.PLAN_ROOM_SEARCH_PATH == self._phase:

                # Extract the numeric location from the room name and set it as the agent's location
                self._agent_loc = int(self._door['room_name'].split()[-1])

                # Store the locations of all area tiles in the current room
                room_tiles = [info['location'] for info in state.values()
                              if 'class_inheritance' in info
                              and 'AreaTile' in info['class_inheritance']
                              and 'room_name' in info
                              and info['room_name'] == self._door['room_name']]
                self._roomtiles = room_tiles

                # Make the plan for searching the area
                self._navigator.reset_full()
                self._navigator.add_waypoints(self._efficientSearch(room_tiles))

                # Initialize variables for storing room victims and switch to following the room search path
                self._room_vics = []
                self._phase = Phase.FOLLOW_ROOM_SEARCH_PATH

            if Phase.FOLLOW_ROOM_SEARCH_PATH == self._phase:
                # Search the area
                self._state_tracker.update(state)
                action = self._navigator.get_move_action(self._state_tracker)
                if action != None:
                    # Identify victims present in the area
                    for info in state.values():
                        if 'class_inheritance' in info and 'CollectableBlock' in info['class_inheritance']:
                            vic = str(info['img_name'][8:-4])
                            # Remember which victim the agent found in this area
                            if vic not in self._room_vics:
                                self._room_vics.append(vic)

                            # Identify the exact location of the victim that was found by the human earlier
                            if vic in self._found_victims and 'location' not in self._found_victim_logs[vic].keys():
                                self._recent_vic = vic
                                # Add the exact victim location to the corresponding dictionary
                                self._found_victim_logs[vic] = {'location': info['location'],
                                                                'room': self._door['room_name'],
                                                                'obj_id': info['obj_id']}
                                if vic == self._goal_vic:
                                    # Communicate which victim was found
                                    self._send_message('Found ' + vic + ' in ' + self._door[
                                        'room_name'] + ' because you told me ' + vic + ' was located here.',
                                                       'RescueBot')
                                    # Add the area to the list with searched areas
                                    if self._door['room_name'] not in self._searched_rooms:
                                        self._searched_rooms.add(self._door['room_name'])
                                    # Do not continue searching the rest of the area but start planning to rescue the victim
                                    self._phase = Phase.FIND_NEXT_GOAL

                            # Identify injured victim in the area
                            if 'healthy' not in vic and vic not in self._found_victims:
                                self._recent_vic = vic
                                # Add the victim and the location to the corresponding dictionary
                                self._found_victims.append(vic)
                                # if it has been reported as collected, but was found again, remove from collected technically
                                # if vic in self._collected_victims:
                                #     self._collected_victims.remove(vic)
                                self._found_victim_logs[vic] = {'location': info['location'],
                                                                'room': self._door['room_name'],
                                                                'obj_id': info['obj_id']}
                                # Communicate which victim the agent found and ask the human whether to rescue the victim now or at a later stage
                                if 'mild' in vic and self._answered == False and not self._waiting:
                                    self._send_message('Found ' + vic + ' in ' + self._door['room_name'] + '. Please decide whether to "Rescue together", "Rescue alone", or "Continue" searching. \n \n \
                                        Important features to consider are: \n safe - victims rescued: ' + str(
                                        self._collected_victims) + '\n explore - areas searched: area ' + str(
                                        self._searched_rooms).replace('area ', '') + '\n \
                                        clock - extra time when rescuing alone: 15 seconds \n afstand - distance between us: ' + self._distance_human,
                                                       'RescueBot')
                                    self.robot_found = True
                                    self._waiting = True

                                # TODO: we can send messages here that the agent has not come yet maybe? or define another phase
                                # that is waiting for person still in which we send a message that we later check in the trust belief method
                                if 'critical' in vic and self._answered == False and not self._waiting:
                                    self._send_message('Found ' + vic + ' in ' + self._door['room_name'] + '. Please decide whether to "Rescue" or "Continue" searching. \n\n \
                                        Important features to consider are: \n explore - areas searched: area ' + str(
                                        self._searched_rooms).replace('area',
                                                                      '') + ' \n safe - victims rescued: ' + str(
                                        self._collected_victims) + '\n \
                                        afstand - distance between us: ' + self._distance_human, 'RescueBot')
                                    self.robot_found = True
                                    self._waiting = True
                                    # Execute move actions to explore the area
                    return action, {}

                # Communicate that the agent did not find the target victim in the area while the human previously communicated the victim was located here
                if self._goal_vic in self._found_victims and self._goal_vic not in self._room_vics and \
                        self._found_victim_logs[self._goal_vic]['room'] == self._door['room_name']:
                    self._send_message(self._goal_vic + ' not present in ' + str(self._door[
                                                                                     'room_name']) + ' because I searched the whole area without finding ' + self._goal_vic + '. If stuck, please decide whether to "Continue" searching.',
                                       'RescueBot')
                    # Remove the victim location from memory
                    self._found_victim_logs.pop(self._goal_vic, None)
                    self._found_victims.remove(self._goal_vic)
                    self._room_vics = []
                    # Reset received messages (bug fix)
                    self.received_messages = []
                    self.received_messages_content = []
                # Add the area to the list of searched areas
                if self._door['room_name'] not in self._searched_rooms:
                    self._searched_rooms.add(self._door['room_name'])
                # Make a plan to rescue a found critically injured victim if the human decides so
                if self.received_messages_content and self.received_messages_content[
                    -1] == 'Rescue' and 'critical' in self._recent_vic:
                    combined_trust = (0.7 * trustBeliefs[self._human_name]['rescue_red']['willingness']
                                      + 0.3 * trustBeliefs[self._human_name]['rescue_red']['competence'])
                    trusting = self.probability_trust(combined_trust)
                    #TODO: consider the time we wait somehow in here, but not quite sure
                    if trusting:
                        self._rescue = 'together'
                        self._answered = True
                        self._waiting = False
                        # Tell the human to come over and help carry the critically injured victim
                        if not state[{'is_human_agent': True}]:
                            self._send_message('Please come to ' + str(self._door['room_name']) + ' to carry ' + str(
                                self._recent_vic) + ' together.', 'RescueBot')
                            self.robot_found = True

                        # Tell the human to carry the critically injured victim together
                        if state[{'is_human_agent': True}]:
                            self._send_message('Lets carry ' + str(
                                self._recent_vic) + ' together! Please wait until I moved on top of ' + str(
                                self._recent_vic) + '.', 'RescueBot')
                        self._goal_vic = self._recent_vic
                        self._recent_vic = None
                        self.robot_found = True
                        print("I am here now cause I trust and should be in the case where robot found:")
                        self._phase = Phase.PLAN_PATH_TO_VICTIM
                    else:
                        self._answered = True
                        self._waiting = False
                        self._send_message('I do not trust you enough to rely on you to save red victim right now'
                                           , 'RescueBot')
                        self._todo.append(self._recent_vic)
                        self._recent_vic = None
                        self._phase = Phase.FIND_NEXT_GOAL

                # Make a plan to rescue a found mildly injured victim together if the human decides so
                if self.received_messages_content and self.received_messages_content[
                    -1] == 'Rescue together' and 'mild' in self._recent_vic:
                    # if we trust: initialize rescue together, else rescue alone
                    combined_trust = (0.7 * trustBeliefs[self._human_name]['rescue_yellow']['willingness']
                                      + 0.3 * trustBeliefs[self._human_name]['rescue_yellow']['competence'])
                    trusting = self.probability_trust(combined_trust)
                    if trusting:
                        self._rescue = 'together'
                        self._answered = True
                        self._waiting = False
                        # Tell the human to come over and help carry the mildly injured victim
                        if not state[{'is_human_agent': True}]:
                            self._send_message('Please come to ' + str(self._door['room_name']) + ' to carry ' + str(
                                self._recent_vic) + ' together.', 'RescueBot')
                        # Tell the human to carry the mildly injured victim together
                        if state[{'is_human_agent': True}]:
                            self._send_message('Lets carry ' + str(
                                self._recent_vic) + ' together! Please wait until I moved on top of ' + str(
                                self._recent_vic) + '.', 'RescueBot')
                        self._goal_vic = self._recent_vic
                        self._recent_vic = None
                        self.robot_found = True
                        self._phase = Phase.PLAN_PATH_TO_VICTIM
                    else:
                        # We decide to save the victim ourselves
                        self._answered = True
                        self._waiting = False
                        self._goal_vic = self._recent_vic
                        self._send_message("Low current trust level; Picking up: " + self._goal_vic, "RescueBot")
                        self._goal_loc = self._remaining[self._goal_vic]
                        self._recent_vic = None
                        self.robot_found = True
                        self._phase = Phase.PLAN_PATH_TO_VICTIM

                # Make a plan to rescue the mildly injured victim alone if the human decides so, and communicate this to the human
                if self.received_messages_content and self.received_messages_content[
                    -1] == 'Rescue alone' and 'mild' in self._recent_vic:
                    # I think we can say that we just do this, no need to look at trust here,
                    # because we trust ourselves to do this
                    self._send_message('Picking up: ' + self._recent_vic + ' in ' + self._door['room_name'] + '.',
                                       'RescueBot')
                    self._rescue = 'alone'
                    self._answered = True
                    self._waiting = False
                    self._goal_vic = self._recent_vic
                    self._goal_loc = self._remaining[self._goal_vic]
                    self._recent_vic = None
                    self.robot_found = True
                    self._phase = Phase.PLAN_PATH_TO_VICTIM
                # Continue searching other areas if the human decides so
                if self.received_messages_content and self.received_messages_content[-1] == 'Continue':
                    combined_trust = (0.7 * trustBeliefs[self._human_name]['rescue_yellow']['willingness']
                                      + 0.3 * trustBeliefs[self._human_name]['rescue_yellow']['competence'])
                    trusting = self.probability_trust(combined_trust)
                    #if we trust, continue, else pick up alone
                    if trusting:
                        self._answered = True
                        self._waiting = False
                        self._todo.append(self._recent_vic)
                        self._recent_vic = None
                        self._phase = Phase.FIND_NEXT_GOAL
                    else:
                        # We decide to save the victim ourselves
                        self._answered = True
                        self._waiting = False
                        self._goal_vic = self._recent_vic
                        self._send_message("Low current trust level; Picking up: " + self._goal_vic, "RescueBot")
                        self._goal_loc = self._remaining[self._goal_vic]
                        self._recent_vic = None
                        self._take_victim_repeat = True # used to enter or ignore the if statement in take victim phase manually
                        self._phase = Phase.PLAN_PATH_TO_VICTIM

                # Remain idle until the human communicates to the agent what to do with the found victim
                if self.received_messages_content and self._waiting and self.received_messages_content[
                    -1] != 'Rescue' and self.received_messages_content[-1] != 'Continue':
                    return None, {}
                # Find the next area to search when the agent is not waiting for an answer from the human or occupied with rescuing a victim
                if not self._waiting and not self._rescue:
                    self._recent_vic = None
                    self._phase = Phase.FIND_NEXT_GOAL
                return Idle.__name__, {'duration_in_ticks': 25}

            if Phase.PLAN_PATH_TO_VICTIM == self._phase:
                # Plan the path to a found victim using its location
                self._navigator.reset_full()
                self._navigator.add_waypoints([self._found_victim_logs[self._goal_vic]['location']])
                # Follow the path to the found victim
                self._phase = Phase.FOLLOW_PATH_TO_VICTIM

            if Phase.FOLLOW_PATH_TO_VICTIM == self._phase:
                # Start searching for other victims if the human already rescued the target victim
                if self._goal_vic and self._goal_vic in self._collected_victims:
                    self._phase = Phase.FIND_NEXT_GOAL

                # Move towards the location of the found victim
                else:
                    self._state_tracker.update(state)

                    action = self._navigator.get_move_action(self._state_tracker)


                    # If there is a valid action, return it; otherwise, switch to taking the victim
                    if action is not None:
                        return action, {}



                    self._phase = Phase.TAKE_VICTIM

            if Phase.TAKE_VICTIM == self._phase:
                # Store all area tiles in a list
                room_tiles = [info['location'] for info in state.values()
                              if 'class_inheritance' in info
                              and 'AreaTile' in info['class_inheritance']
                              and 'room_name' in info
                              and info['room_name'] == self._found_victim_logs[self._goal_vic]['room']]
                self._roomtiles = room_tiles
                objects = []

                robot_found_victim = self.robot_found

                # TODO: Here we need to incorporate the time to wait for human to come to and if he is there when we come
                # When the victim has to be carried by human and agent together, check whether human has arrived at the victim's location
                for info in state.values():

                    # When the victim has to be carried by human and agent together, check whether human has arrived at the victim's location
                    if 'class_inheritance' in info and 'CollectableBlock' in info['class_inheritance'] and 'critical' in \
                            info['obj_id'] and info['location'] in self._roomtiles or \
                            'class_inheritance' in info and 'CollectableBlock' in info[
                        'class_inheritance'] and 'mild' in info['obj_id'] and info[
                        'location'] in self._roomtiles and self._rescue == 'together' or \
                            self._goal_vic in self._found_victims and self._goal_vic in self._todo and len(
                        self._searched_rooms) == 0 and 'class_inheritance' in info and 'CollectableBlock' in info[
                        'class_inheritance'] and 'critical' in info['obj_id'] and info['location'] in self._roomtiles or \
                            self._goal_vic in self._found_victims and self._goal_vic in self._todo and len(
                        self._searched_rooms) == 0 and 'class_inheritance' in info and 'CollectableBlock' in info[
                        'class_inheritance'] and 'mild' in info['obj_id'] and info['location'] in self._roomtiles:

                        objects.append(info)

                        # Remain idle when the human has not arrived at the location
                        if self.robot_found:
                            if self._human_name not in info['name']:
                                self._waiting = True
                                self._moving = False
                                return None, {}
                        else:
                            if self._human_name not in info['name'] and not self._take_victim_repeat:
                                # TODO: time it takes for human to arrive if we have called him, this needs more testing
                                #  for loops and normal performance

                                if 'mild' in info['obj_id']:
                                    combined_trust = (0.5 * trustBeliefs[self._human_name]['rescue_yellow']['willingness']
                                                      + 0.5 * trustBeliefs[self._human_name]['rescue_yellow']['competence'])
                                    we_trust = self.probability_trust(combined_trust)
                                    self._send_message('Human agent not present at location to save mild victim together', "RescueBot")
                                else:
                                    combined_trust = (0.5 * trustBeliefs[self._human_name]['rescue_red']['willingness']
                                                      + 0.5 * trustBeliefs[self._human_name]['rescue_red']['competence'])
                                    we_trust = self.probability_trust(combined_trust)
                                    self._send_message('Human agent not present at location to save critical victim together', "RescueBot")

                                if we_trust:
                                    self._waiting = True
                                    self._moving = False
                                    self._send_message("Waiting for human to come pick up victim together",
                                                       "RescueBot")
                                    return None, {}

                                else:
                                    if 'mild' in info['obj_id']:
                                        self._answered = True
                                        self._waiting = False
                                        self._goal_vic = ' '.join(info['obj_id'].split('_')[:3])
                                        self._send_message("Human not present will not wait; Picking up: " + self._goal_vic,
                                                           "RescueBot")
                                        self._goal_loc = self._remaining[self._goal_vic]
                                        self._recent_vic = None
                                        self._phase = Phase.PLAN_PATH_TO_VICTIM
                                        self._take_victim_repeat = True

                                    else:
                                        self._take_victim_repeat = False
                                        self._waiting = False
                                        self._recent_vic = None
                                        self._goal_vic = None
                                        self._phase = Phase.PICK_UNSEARCHED_ROOM
                                        self._send_message("Human not present will not wait; Proceeding to next goal",
                                                           "RescueBot")
                self.robot_found = False

                if self._take_victim_repeat:
                    self._rescue = 'alone'
                self._take_victim_repeat = False


                # Add the victim to the list of rescued victims when it has been picked up
                if self._goal_vic and len(objects) == 0 and 'critical' in self._goal_vic or len(
                        objects) == 0 and 'mild' in self._goal_vic and self._rescue == 'together':
                    self._waiting = False
                    if self._goal_vic not in self._collected_victims:
                        self._collected_victims.append(self._goal_vic)
                    self._carrying_together = True
                    # Determine the next victim to rescue or search
                    self._phase = Phase.FIND_NEXT_GOAL
                # When rescuing mildly injured victims alone, pick the victim up and plan the path to the drop zone

                if self._goal_vic and 'mild' in self._goal_vic and self._rescue == 'alone':
                    self._phase = Phase.PLAN_PATH_TO_DROPPOINT
                    if self._goal_vic not in self._collected_victims:
                        self._collected_victims.append(self._goal_vic)
                    self._carrying = True
                    return CarryObject.__name__, {'object_id': self._found_victim_logs[self._goal_vic]['obj_id'],
                                                  'human_name': self._human_name}

            if Phase.PLAN_PATH_TO_DROPPOINT == self._phase:
                self._navigator.reset_full()
                # Plan the path to the drop zone
                self._navigator.add_waypoints([self._goal_loc])
                # Follow the path to the drop zone
                self._phase = Phase.FOLLOW_PATH_TO_DROPPOINT

            if Phase.FOLLOW_PATH_TO_DROPPOINT == self._phase:
                # Communicate that the agent is transporting a mildly injured victim alone to the drop zone
                if 'mild' in self._goal_vic and self._rescue == 'alone':
                    self._send_message('Transporting ' + self._goal_vic + ' to the drop zone.', 'RescueBot')
                self._state_tracker.update(state)
                # Follow the path to the drop zone
                action = self._navigator.get_move_action(self._state_tracker)
                if action is not None:
                    return action, {}
                # Drop the victim at the drop zone
                self._phase = Phase.DROP_VICTIM

            if Phase.DROP_VICTIM == self._phase:
                # Communicate that the agent delivered a mildly injured victim alone to the drop zone
                if 'mild' in self._goal_vic and self._rescue == 'alone':
                    self._send_message('Delivered ' + self._goal_vic + ' at the drop zone.', 'RescueBot')
                # Identify the next target victim to rescue
                self._phase = Phase.FIND_NEXT_GOAL
                self._rescue = None
                self._current_door = None
                self._tick = state['World']['nr_ticks']
                self._carrying = False
                # Drop the victim on the correct location on the drop zone
                return Drop.__name__, {'human_name': self._human_name}

            if Phase.WAIT_FOR_HUMAN == self._phase:
                print("Chackam nehranimajkoto veche chas")


    def probability_trust(self, trust_value):
        prob_trust = (trust_value + 1) / 2
        prob_trust = np.clip(prob_trust, 0.01, 1)
        # Generate a single random True/False
        trust_state = np.random.choice([True, False], p=[prob_trust, 1 - prob_trust])
        return trust_state

    def _get_drop_zones(self, state):
        '''
        @return list of drop zones (their full dict), in order (the first one is the
        place that requires the first drop)
        '''
        places = state[{'is_goal_block': True}]
        places.sort(key=lambda info: info['location'][1])
        zones = []
        for place in places:
            if place['drop_zone_nr'] == 0:
                zones.append(place)
        return zones

    def _process_messages(self, state, teamMembers, condition):
        '''
        process incoming messages received from the team members
        '''
        # Load the most recent trust beliefs in the current game round
        trustBeliefs = self._loadBelief(self._team_members, self._folder + '/beliefs/currentTrustBelief.csv')

        receivedMessages = {}
        # Create a dictionary with a list of received messages from each team member
        for member in teamMembers:
            receivedMessages[member] = []
        for mssg in self.received_messages:
            for member in teamMembers:
                if mssg.from_id == member:
                    receivedMessages[member].append(mssg.content)
        # Check the content of the received messages
        for mssgs in receivedMessages.values():
            for msg in mssgs:
                # If a received message involves team members searching areas, add these areas to the memory of areas that have been explored
                if msg.startswith("Search:"):
                    area = 'area ' + msg.split()[-1]
                    # If you trust the human enough, mark this room as searched, otherwise better search it yourself
                    if trustBeliefs[self._human_name]['search'][
                        'competence'] > 0.6 and area not in self._searched_rooms:
                        self._searched_rooms.add(area)
                # If a received message involves team members finding victims, add these victims and their locations to memory
                if msg.startswith("Found:"):
                    # Identify which victim and area it concerns
                    if len(msg.split()) == 6:
                        foundVic = ' '.join(msg.split()[1:4])
                    else:
                        foundVic = ' '.join(msg.split()[1:5])
                    loc = 'area ' + msg.split()[-1]

                    # If you trust the human enough, mark this room as searched, otherwise better search it yourself
                    if trustBeliefs[self._human_name]['search']['competence'] > 0.6 and loc not in self._searched_rooms:
                        self._searched_rooms.add(loc)

                    # Add the victim and its location to memory
                    if foundVic not in self._found_victims:
                        self._found_victims.append(foundVic)
                        self._found_victim_logs[foundVic] = {'room': loc}
                    elif self._found_victim_logs[foundVic]['room'] != loc and 'location' not in self._found_victim_logs[
                        foundVic]:
                        # Only change the room location if the agent has not already found the victim on its own
                        self._found_victim_logs[foundVic] = {'room': loc}

                    # Decide to help the human carry a found victim when the human's condition is 'weak'
                    if condition == 'weak':
                        self._rescue = 'together'
                    # Add the found victim to the to do list when the human's condition is not 'weak'
                    if 'mild' in foundVic and condition != 'weak':
                        self._todo.append(foundVic)
                # If a received message involves team members rescuing victims, add these victims and their locations to memory
                if msg.startswith('Collect:'):
                    # Identify which victim and area it concerns
                    if len(msg.split()) == 6:
                        collectVic = ' '.join(msg.split()[1:4])
                    else:
                        collectVic = ' '.join(msg.split()[1:5])
                    loc = 'area ' + msg.split()[-1]
                    # Add the area to the memory of searched areas
                    if loc not in self._searched_rooms:
                        #this does not work due to the threshold for trusting in search
                        # self._send_message("Victim was not communicated as found", "RescueBot")
                        self._searched_rooms.add(loc)
                    # Add the victim and location to the memory of found victims
                    if collectVic not in self._found_victims:
                        #this does not work due to the threshold for trusting in search
                        # self._send_message("Victim was not communicated as found", "RescueBot")
                        self._found_victims.append(collectVic)
                        self._found_victim_logs[collectVic] = {'room': loc}
                    if collectVic in self._found_victims and self._found_victim_logs[collectVic]['room'] != loc:
                        self._found_victim_logs[collectVic] = {'room': loc}
                    # Add the victim to the memory of rescued victims when the human's condition is not weak
                    if condition != 'weak' and collectVic not in self._collected_victims:
                        self._collected_victims.append(collectVic)
                    # Decide to help the human carry the victim together when the human's condition is weak
                    if condition == 'weak':
                        self._rescue = 'together'
                # If a received message involves team members asking for help with removing obstacles, add their location to memory and come over
                if msg.startswith('Remove:'):
                    # Come over immediately when the agent is not carrying a victim
                    if not self._carrying:
                        # Identify at which location the human needs help
                        area = 'area ' + msg.split()[-1]
                        self._door = state.get_room_doors(area)[0]
                        self._doormat = state.get_room(area)[-1]['doormat']
                        if area in self._searched_rooms:
                            self._searched_rooms.remove(area)
                        # Clear received messages (bug fix)
                        self.received_messages = []
                        self.received_messages_content = []
                        self._moving = True
                        self._remove = True
                        if self._waiting and self._recent_vic:
                            self._todo.append(self._recent_vic)
                        self._waiting = False
                        # Let the human know that the agent is coming over to help
                        self._send_message(
                            'Moving to ' + str(self._door['room_name']) + ' to help you remove an obstacle.',
                            'RescueBot')
                        # Plan the path to the relevant area
                        self._phase = Phase.PLAN_PATH_TO_ROOM
                    # Come over to help after dropping a victim that is currently being carried by the agent
                    else:
                        area = 'area ' + msg.split()[-1]
                        self._send_message('Will come to ' + area + ' after dropping ' + self._goal_vic + '.',
                                           'RescueBot')
            # Store the current location of the human in memory
            if mssgs and mssgs[-1].split()[-1] in ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13',
                                                   '14']:
                self._human_loc = int(mssgs[-1].split()[-1])

    def _loadBelief(self, members, filepath):
        """
        Loads trust belief values if agent already collaborated with human before, otherwise trust belief values are initialized using default values.
        """
        # Create a dictionary with trust values for all team members
        trustBeliefs = {}
        # Set a default starting trust value
        trustfile_header = []
        trustfile_contents = []
        # Check if agent already collaborated with this human before, if yes: load the corresponding trust values, if no: initialize using default trust values
        with open(filepath) as csvfile:
            reader = csv.reader(csvfile, delimiter=';', quotechar="'")
            for row in reader:
                if trustfile_header == []:
                    trustfile_header = row
                    continue
                # Retrieve trust values 
                if row and row[0] == self._human_name:
                    name = row[0]
                    task = row[1]
                    competence = float(row[2])
                    willingness = float(row[3])
                    if name not in trustBeliefs:
                        trustBeliefs[name] = {}

                    trustBeliefs[name][task] = {'competence': competence, 'willingness': willingness}

            # Initialize default trust values
            if self._human_name not in trustBeliefs:
                trustBeliefs[self._human_name] = {}

                competence = self._default_trust_value
                willingness = self._default_trust_value
                trustBeliefs[self._human_name]['search'] = {'competence': competence, 'willingness': willingness}
                trustBeliefs[self._human_name]['rescue_red'] = {'competence': competence,
                                                                'willingness': willingness}
                trustBeliefs[self._human_name]['rescue_yellow'] = {'competence': competence,
                                                                   'willingness': willingness}
        return trustBeliefs

    def _trustBelief(self, members, trustBeliefs, folder, receivedMessages):
        '''
        Baseline implementation of a trust belief. Creates a dictionary with trust belief scores for each team member, for example based on the received messages.
        '''
        # for some reason values for the trust for rescue are not updated properly for red
        # and I think for yellow, they drop to -1 really fast for some reason, need to look into that
        # I think we are probably updating them sometimes when we should not be updating them, error in the
        # if statements again possibly
        # if len(receivedMessages) == 0:
        #     return
        tick_nr = self._state['World']['nr_ticks']
        _distance_drop = 'far'
        if self._agent_loc in [1, 2, 5, 6, 8, 9, 11, 12]:
            _distance_drop = 'far'
        if self._agent_loc in [3, 4, 7, 10, 13, 14]:
            _distance_drop = 'close'
        self._tick_distance_goal[tick_nr] = _distance_drop

        diffLen = len(receivedMessages) + self._last_length_received_messages - len(self._current_tick_received_messages)
        new_messages = receivedMessages[-diffLen:] if diffLen > 0 else []
        for message in new_messages:
            self._current_tick_received_messages.append((message, tick_nr))

        diffLen_send = len(self._send_messages) + self._last_length_send_messages - len(self._send_message_ticks)
        new_messages = self._send_messages[-diffLen_send:] if diffLen_send > 0 else []
        for message in new_messages:
            self._send_message_ticks.append((message, tick_nr))

        # we can use the time it takes to do it alone if applicable to be the time to wait/correspond to it
        response_time_threshold = 100
        relevant_response_time_threshold_yellow = 100
        relevant_response_time_threshold_red = 100
        given_relevant_response_in_time_red = False
        given_relevant_response_in_time_yellow = False
        claimed_saved = []

        send_index = 0
        received_index = 0
        for i in range(0, len(self._current_tick_received_messages) + len(self._send_message_ticks)):
            if i < len(self._send_message_ticks):
                msg = self._send_message_ticks[i][0];
                if 'Human agent not present at location to save mild victim together' == msg:
                    print("Gave up on task")
                    trustBeliefs[self._human_name]['rescue_yellow']['willingness'] -= 0.2
                    trustBeliefs[self._human_name]['rescue_yellow']['competence'] -= 0.2
                elif 'Human agent not present at location to save critical victim together' == msg:
                    print("Gave up on critical task")
                    trustBeliefs[self._human_name]['rescue_red']['willingness'] -= 0.2
                    trustBeliefs[self._human_name]['rescue_red']['competence'] -= 0.2
                elif 'Victim was not communicated as found' in msg:
                    print("Did not follow sequence of tasks")
                    trustBeliefs[self._human_name]['rescue_yellow']['competence'] -= 0.05
                    trustBeliefs[self._human_name]['rescue_yellow']['willingness'] += 0.1


            if send_index == len(self._send_message_ticks):
                cur_message = self._current_tick_received_messages[received_index][0]
                received_index += 1

            elif received_index == len(self._current_tick_received_messages):
                cur_message = self._send_message_ticks[send_index][0]
                send_index += 1
            else:
                if self._send_message_ticks[send_index][1] <= self._current_tick_received_messages[received_index][1]:
                    cur_message = self._send_message_ticks[send_index][0]
                    send_index += 1
                else:
                    cur_message = self._current_tick_received_messages[received_index][0]
                    received_index += 1

            self.match_collect(cur_message, claimed_saved, trustBeliefs)
            self.match_picking_up(cur_message, claimed_saved, trustBeliefs)

            self.check_if_collected_victim_found(cur_message, claimed_saved, trustBeliefs)

        for send_message, send_tick in [t for t in self._send_message_ticks if
                                        t[0].startswith('Found') and 'injured' in t[0] and 'blocking' not in t[0]]:


            next_received_messages = self.find_next_received(send_tick)
            for response, resp_tick in next_received_messages:

                if 'Found mild' in send_message:
                    if not given_relevant_response_in_time_yellow:
                        # Extract "distance between us"
                        distance_match = re.search(r'distance between us: (\w+)', send_message)
                        distance_human = distance_match.group(1) if distance_match else "far"

                        if response in self._lookup_table['Mild'].keys():
                            given_relevant_response_in_time_yellow = True
                            print('relevant response for mild to updated yellow values')
                            trustBeliefs[self._human_name]['rescue_yellow']['competence'] += \
                                self._lookup_table["Mild"][response][
                                    (distance_human, self._tick_distance_goal[send_tick])][
                                    'competence']
                            trustBeliefs[self._human_name]['rescue_yellow']['willingness'] += \
                                self._lookup_table["Mild"][response][
                                    (distance_human, self._tick_distance_goal[send_tick])][
                                    'willingness']
                        else:
                            print('not relevant response for mild to updated yellow values after tome')
                            if resp_tick - send_tick > relevant_response_time_threshold_yellow:
                                trustBeliefs[self._human_name]['rescue_yellow']['willingness'] -= 0.1
                                trustBeliefs[self._human_name]['rescue_yellow']['competence'] -= 0.1

                elif 'Found critical' in send_message:
                    if not given_relevant_response_in_time_red:

                        victims_match = re.search(r'safe - victims rescued: \[(.*?)\]', send_message)
                        victims = victims_match.group(1).split(", ") if victims_match else []

                        # Remove empty strings if no victims are present
                        victims = [v for v in victims if v]

                        distance_match = re.search(r'distance between us: (\w+)', send_message)
                        distance_human = distance_match.group(1) if distance_match else "far"
                        if len(victims) > 5 and 'Continue' == response:
                            trustBeliefs[self._human_name]['rescue_red']['competence'] -= 0.1
                            trustBeliefs[self._human_name]['rescue_red']['willingness'] -= 0.1

                        if response in self._lookup_table['Critical'].keys():
                            print('relevant response for critical to updated red values')
                            given_relevant_response_in_time_red = True
                            trustBeliefs[self._human_name]['rescue_red']['competence'] += \
                                self._lookup_table["Critical"][response][
                                    (distance_human, self._tick_distance_goal[send_tick])]['competence']
                            trustBeliefs[self._human_name]['rescue_red']['willingness'] += \
                                self._lookup_table["Critical"][response][
                                    (distance_human, self._tick_distance_goal[send_tick])]['willingness']
                        else:
                            print('not relevant response for critical to updated red values after time')
                            if resp_tick - send_tick > relevant_response_time_threshold_red:
                                trustBeliefs[self._human_name]['rescue_red']['competence'] -= 0.1
                                trustBeliefs[self._human_name]['rescue_red']['willingness'] -= 0.1

        for rec_message, rec_tick in [t for t in self._current_tick_received_messages if 'Found:' in t[0]]:
            next_received_messages = self.find_next_received(rec_tick)
            if 'mild' in rec_message:
                regex_extractor = re.search(r"Found: (.*?) in (\d+)", message)
                found_victim = regex_extractor.group(1)
                next_message_length = 2 if (len(next_received_messages) >= 2) else len(next_received_messages)
                collected_victim = False
                if next_message_length>1:
                    for i in range(0, next_message_length):
                        if 'Collect' in next_received_messages[i][0]:
                            regex_extractor = re.search(r"Collect: (.*?) in (\d+)", message)
                            collected = regex_extractor.group(1)
                            if collected == found_victim:
                                print('collect in response threshold')
                                collected_victim = True
                                trustBeliefs[self._human_name]['rescue_yellow']['competence'] += 0.2
                                trustBeliefs[self._human_name]['rescue_yellow']['willingness'] += 0.2
                    if not collected_victim:
                        print('not collect in response threshold')
                        trustBeliefs[self._human_name]['rescue_yellow']['competence'] -= 0.1
                        trustBeliefs[self._human_name]['rescue_yellow']['willingness'] -= 0.1

        self.trustBeliefSearch(receivedMessages, trustBeliefs)

        # Restrict the competence and willingness beliefs to a range of -1 to 1
        trustBeliefs[self._human_name]['search']['competence'] = np.clip(
            trustBeliefs[self._human_name]['search']['competence'], -1, 1)
        trustBeliefs[self._human_name]['search']['willingness'] = np.clip(
            trustBeliefs[self._human_name]['search']['willingness'], -1, 1)

        trustBeliefs[self._human_name]['rescue_red']['competence'] = np.clip(
            trustBeliefs[self._human_name]['rescue_red']['competence'], -1, 1)
        trustBeliefs[self._human_name]['rescue_red']['willingness'] = np.clip(
            trustBeliefs[self._human_name]['rescue_red']['willingness'], -1, 1)

        trustBeliefs[self._human_name]['rescue_yellow']['competence'] = np.clip(
            trustBeliefs[self._human_name]['rescue_yellow']['competence'], -1, 1)
        trustBeliefs[self._human_name]['rescue_yellow']['willingness'] = np.clip(
            trustBeliefs[self._human_name]['rescue_yellow']['willingness'], -1, 1)

        # Save current trust belief values so we can later use and retrieve them to add to a csv file with all the logged trust belief values
        with open(folder + '/beliefs/currentTrustBelief.csv', mode='w') as csv_file:
            csv_writer = csv.writer(csv_file, delimiter=';', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            csv_writer.writerow(['name', 'task', 'competence', 'willingness'])
            csv_writer.writerow([self._human_name, 'search', trustBeliefs[self._human_name]['search']['competence'],
                                 trustBeliefs[self._human_name]['search']['willingness']])
            csv_writer.writerow([self._human_name, 'rescue_red',
                                 trustBeliefs[self._human_name]['rescue_red']['competence'],
                                 trustBeliefs[self._human_name]['rescue_red']['willingness']])
            csv_writer.writerow([self._human_name, 'rescue_yellow',
                                 trustBeliefs[self._human_name]['rescue_yellow']['competence'],
                                 trustBeliefs[self._human_name]['rescue_yellow']['willingness']])

        return trustBeliefs

    def find_next_received(self, tick):
        return [t for t in self._current_tick_received_messages if t[1] > tick]

    def check_if_collected_victim_found(self, send_message, claimed_saved, trustBeliefs):
        match = re.search(r'Found\s+(\w+)\s+(\w+)\s+(\w+)', send_message)
        victim = None
        if match:
            victim = " ".join(match.groups()).replace(':', '')  # Join the words into a single string
            if victim in claimed_saved:
                trustBeliefs[self._human_name]['rescue_yellow']['competence'] -= 0.3
                trustBeliefs[self._human_name]['rescue_yellow']['willingness'] -= 0.3

    def match_collect(self,send_message, claimed_saved, trustBeliefs):
        match = re.search(r'Collect:\s+(\w+)\s+(\w+)\s+(\w+)', send_message)
        self.update_collected_victims(claimed_saved, trustBeliefs, match)

    def match_picking_up(self,send_message, claimed_saved, trustBeliefs):
        match = re.search(r'Picking up:\s+(\w+)\s+(\w+)\s+(\w+)', send_message)
        self.update_collected_victims(claimed_saved, trustBeliefs, match)

    def update_collected_victims(self, claimed_saved, trustBeliefs, match):
        #I think this needs to be updated to facilitate for the victims that the robot picks up
        #I do not think this is happening right now, also new messages for picking up victims
        if not match:
            return

        victim = " ".join(match.groups())  # Join the words into a single string
        #this needs to be here so that victim is not empty if there is a match only, otherwise,
        # it will happen all the time
        if victim not in claimed_saved:
            claimed_saved.append(victim)
        else:
            trustBeliefs[self._human_name]['rescue_yellow']['competence'] -= 0.1
            trustBeliefs[self._human_name]['rescue_yellow']['willingness'] -= 0.1


    def trustBeliefSearch(self, receivedMessages, trustBeliefs):
        # Update the trust value based on for example the received messages
        # Since `receivedMessages` is a list of all messages,
        # some messages will be counted many times, so we reset the trust values to defaults here.

        # trustBeliefs[self._human_name]['search']['competence'] = self._default_trust_value
        # trustBeliefs[self._human_name]['search']['willingness'] = self._default_trust_value
        #print(f"Current belief: {trustBeliefs[self._human_name]['search']['competence']}, {trustBeliefs[self._human_name]['search']['willingness']}")

        area_rec_messages = {}
        area_sent_messages = {}
        for room in range(1, 15):
            area_rec_messages[room] = []
            area_sent_messages[room] = []
        for message in receivedMessages:
            if message.startswith('Search:'):
                area = message.split()[-1]
                area_rec_messages[int(area)].append(message)
            elif message.startswith('Found:'):  # found victim
                regex_extractor = re.search(r"Found: (.*?) in (\d+)", message)
                victim = regex_extractor.group(1)
                room_number = int(regex_extractor.group(2))
                area_rec_messages[room_number].append(message)
                # Task: Search: Communicate findings ("I have found")
                if victim in self._found_victims and 'area ' + str(room_number) == self._found_victim_logs[victim][
                    'room']:
                    # the human communicated a possible room location of the victim
                    # TODO use some sort of confidence level (multiplying factor) such that when the human communicates correctly multiple times, the confidence increases
                    # e.g. trust_factor and distrust_factor between 0 and 1
                    trustBeliefs[self._human_name]['search']['willingness'] += 0.10

                    if 'location' in self._found_victim_logs[victim]:
                        # the human was right and the agent found the human in that exact room
                        trustBeliefs[self._human_name]['search']['competence'] += 0.50
                else:
                    # the human lied (the agent could not find the victim in that room/ the agent found the victim in another room/ the human communicated multiple rooms for the same victim)
                    trustBeliefs[self._human_name]['search']['willingness'] -= 0.50

            elif message.startswith('Collect:'):  # rescue victim
                regex_extractor = re.search(r"Collect: (.*?) in (\d+)", message)
                room_number = int(regex_extractor.group(2))
                area_rec_messages[room_number].append(message)
            elif message.startswith('Remove:'):  # remove obstacle
                regex_extractor = re.search(r"Remove: at (\d+)", message)
                room_number = int(regex_extractor.group(1))
                area_rec_messages[room_number].append(message)
        for message in self._send_messages:
            if message.startswith('Moving to'):  # moving to area
                regex_extractor = re.search(r"Moving to area (\d+)", message)
                room_number = int(regex_extractor.group(1))
                area_sent_messages[room_number].append(message)
            elif message.startswith('Found ') and 'blocking' in message:  # obstacle in area
                regex_extractor = re.search(r"Found (.*?) blocking area (\d+)", message)
                room_number = int(regex_extractor.group(2))
                area_sent_messages[room_number].append(message)
            elif message.startswith('Found '):
                regex_extractor = re.search(r"Found (.*?) in area (\d+)", message)  # victim in area
                room_number = int(regex_extractor.group(2))
                area_sent_messages[room_number].append(message)
        for room in area_rec_messages:
            # If the human searched the area:
            human_searched_area = any('Search' in msg for msg in area_rec_messages[room])
            if human_searched_area:

                human_reported_obstacle = any('Remove' in msg for msg in area_rec_messages[room])
                robot_found_obstacle = any('Found' in msg and 'blocking' in msg for msg in area_sent_messages[room])

                # If the robot finds an obstacle not called out by the human
                if robot_found_obstacle and not human_reported_obstacle:
                    trustBeliefs[self._human_name]['search']['competence'] -= 0.3  # Human failed to report an obstacle
                    #print(f"Didn't report obstacle{room}")
                elif human_reported_obstacle:
                    trustBeliefs[self._human_name]['search'][
                        'competence'] += 0.1  # Human correctly reported an obstacle

                human_reported_victim = any('Found' in msg and not 'blocking' in msg for msg in area_rec_messages[room])
                robot_found_victim = any('Found' in msg and 'in' in msg for msg in area_sent_messages[room])
                human_rescued_victim = any('Collect' in msg for msg in area_rec_messages[room])

                if robot_found_victim and not (human_reported_victim or human_rescued_victim):
                    trustBeliefs[self._human_name]['search'][
                        'competence'] -= 0.3  # Human failed to report or rescue a victim
                    #print(f"Didn't report or rescue victim {room}")
                elif human_reported_victim or human_rescued_victim:
                    trustBeliefs[self._human_name]['search'][
                        'competence'] += 0.1  # Human correctly reported or rescued a victim

                # If nothing was announced and nothing was found, increase competence
                if (not robot_found_victim and not robot_found_obstacle
                        and not human_reported_victim and not human_reported_obstacle
                        and not human_rescued_victim):
                    trustBeliefs[self._human_name]['search']['competence'] += 0.1
        #print(
        #    f"Updated belief: {trustBeliefs[self._human_name]['search']['competence']}, {trustBeliefs[self._human_name]['search']['willingness']}")

    def _send_message(self, mssg, sender):
        """
        send messages from agent to other team members
        """
        msg = Message(content=mssg, from_id=sender)
        if msg.content not in self.received_messages_content and 'Our score is' not in msg.content:
            self.send_message(msg)
            self._send_messages.append(msg.content)
        # Sending the hidden score message (DO NOT REMOVE)
        if 'Our score is' in msg.content:
            self.send_message(msg)

    def _getClosestRoom(self, state, objs, currentDoor):
        """
        calculate which area is closest to the agent's location
        """
        agent_location = state[self.agent_id]['location']
        locs = {}
        for obj in objs:
            locs[obj] = state.get_room_doors(obj)[0]['location']
        dists = {}
        for room, loc in locs.items():
            if currentDoor != None:
                dists[room] = utils.get_distance(currentDoor, loc)
            if currentDoor == None:
                dists[room] = utils.get_distance(agent_location, loc)

        return min(dists, key=dists.get)

    def _efficientSearch(self, tiles):
        '''
        efficiently transverse areas instead of moving over every single area tile
        '''
        x = []
        y = []
        for i in tiles:
            if i[0] not in x:
                x.append(i[0])
            if i[1] not in y:
                y.append(i[1])
        locs = []
        for i in range(len(x)):
            if i % 2 == 0:
                locs.append((x[i], min(y)))
            else:
                locs.append((x[i], max(y)))
        return locs

    # def random_pick(self, competence):
    #     random_number = random.random()
    #     if random_number < (competence+1)/2:
    #         return True
    #     else:
    #         return False
