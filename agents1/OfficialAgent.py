import csv
import enum
import math
import random
import os
from dataclasses import dataclass
from typing import Optional

import numpy as np
from matrx import utils
from matrx.actions.object_actions import RemoveObject
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


@dataclass
class Objective:
    action: str

    start_time: int

    area: Optional[int] = None

    person: Optional[int] = None

    end_time: Optional[int] = None


baseline = None # To change depending on the evaluation method
# Can be set to "NEVER-TRUST", "ALWAYS-TRUST" or "RANDOM-TRUST"

random_competence : float = float(random.uniform(-1, 1))
random_willingness : float = float(random.uniform(-1, 1))

class BaselineAgent(ArtificialBrain):
    def __init__(self, slowdown, condition, name, folder):
        super().__init__(slowdown, condition, name, folder)
        # Initialization of some relevant variables
        self._tick = 0
        self._slowdown = slowdown
        self._condition = condition
        self._human_name = name
        self._folder = folder
        self._phase = Phase.INTRO
        self._room_vics = []
        self._searched_rooms = []
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

        self._atomic_actions = ['Search', 'Collect', 'Found', "Remove"]

        self._aid_remove = False
        self._obstacle_is_tree = False

        # idle time
        self.idle_since = None
        self._last_processed_message = 0

        # Define competence and willingness thresholds
        self._competence_threshold = -0.1
        self._willingness_threshold = -0.1

        self._objectiveHistory: dict[str, list[Objective]] = {}  # Group by possible action
        self._dictionary_to_print: dict[int, dict[str, float]] = {}

        self.distances = {
            'close': 0,  # +0 seconds
            'medium': 10,  # +1 seconds
            'far': 30 # +3 seconds
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
        self._tick += 1
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
        messages_to_process = self._received_messages[self._last_processed_message + 1:]
        self._process_messages(state, self._team_members, self._condition)
        # Initialize and update trust beliefs for team members
        trustBeliefs = self._loadBelief(self._team_members, self._folder, baseline)
        self._trustBelief(self._tick, self._team_members, trustBeliefs, self._folder, messages_to_process, state, baseline)
        self._last_processed_message = len(self._received_messages) - 1

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

            willingness = trustBeliefs.get(self._human_name).get('willingness')
            competence = trustBeliefs.get(self._human_name).get('competence')

            if baseline is None:
                pass
            elif baseline == "NEVER-TRUST":
                willingness = -1
                competence = -1
            elif baseline == "ALWAYS-TRUST":
                willingness = 1
                competence = 1
            else:
                willingness = random_willingness
                competence = random_competence

            if Phase.INTRO == self._phase:
                # Send introduction message
                self._send_message('Hello! My name is RescueBot. Together we will collaborate and try to search and rescue the 8 victims on our right as quickly as possible. \
                Each critical victim (critically injured girl/critically injured elderly woman/critically injured man/critically injured dog) adds 6 points to our score, \
                each mild victim (mildly injured boy/mildly injured elderly man/mildly injured woman/mildly injured cat) 3 points. \
                If you are ready to begin our mission, you can simply start moving.', 'RescueBot')
                # Wait untill the human starts moving before going to the next phase, otherwise remain idle
                if not state[{'is_human_agent': True}]:
                    self._phase = Phase.FIND_NEXT_GOAL
                else:
                    return None, {}

            if Phase.FIND_NEXT_GOAL == self._phase: # TODO: Here
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
                        # Rescue together when victim is critical or when the human is weak and the victim is mildly injured and unwilling
                        if 'critical' in vic or 'mild' in vic and self._condition == 'weak' and willingness < self._willingness_threshold:
                            self._rescue = 'together'
                        # Rescue alone if the victim is mildly injured and the human not weak and unwilling
                        if 'mild' in vic and self._condition != 'weak' and willingness < self._willingness_threshold:
                            self._rescue = 'alone'
                        # Plan path to victim because the exact location is known (i.e., the agent found this victim)
                        if 'location' in self._found_victim_logs[vic].keys():
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
                    self._searched_rooms = []
                    self._send_messages = []
                    self.received_messages = []
                    self.received_messages_content = []
                    self._send_message('Going to re-search all areas.', 'RescueBot')
                    print("Re-search again so willingness goes down")
                    willingness -= 0.4
                    self._phase = Phase.FIND_NEXT_GOAL # TODO: HERE
                # If there are still areas to search, define which one to search next
                else:
                    # Identify the closest door when the agent did not search any areas yet
                    if self._current_door == None:
                        # Find all area entrance locations
                        self._door = \
                        state.get_room_doors(self._getClosestRoom(state, unsearched_rooms, agent_location))[
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
                            # Determine the next area to explore if the human tells the agent not to remove the obstacle
                        if self.received_messages_content and self.received_messages_content[
                            -1] == 'Continue' and not self._remove:
                            self._answered = True
                            self._waiting = False
                            self.idle_since = None
                            # Add area to the to do list
                            self._to_search.append(self._door['room_name'])
                            self._phase = Phase.FIND_NEXT_GOAL
                        # Wait for the human to help removing the obstacle and remove the obstacle together
                        if self.received_messages_content and self.received_messages_content[
                            -1] == 'Remove' or self._remove:
                            if not self._remove:
                                self._answered = True
                            # Tell the human to come over and be idle until human arrives
                            if not state[{'is_human_agent': True}]:
                                self._send_message(
                                    'Please come to ' + str(self._door['room_name']) + ' to remove rock.',
                                    'RescueBot')
                                if self.idle_since is None:
                                    self.idle_since = self._tick
                                return None, {}
                            # Tell the human to remove the obstacle when he/she arrives
                            if state[{'is_human_agent': True}]:
                                self._send_message('Lets remove rock blocking ' + str(self._door['room_name']) + '!',
                                                   'RescueBot')
                                self.idle_since = None
                                return None, {}
                        # Remain idle until the human communicates what to do with the identified obstacle
                        else:
                            if self.idle_since is not None and self._tick - self.idle_since > self._calculate_timeout(
                                    self._loadBelief(self._team_members, self._folder, baseline), 50): # Updates timeout based on expected time of removal
                                competence -= self._calculate_competence_update(trustBeliefs, 0.2 if baseline is None else 0.0)

                                trustBeliefs[self._human_name]["willingness"] = willingness
                                trustBeliefs[self._human_name]['competence'] = competence
                                self._update_csv(competence, willingness)
                                print("UPDATE: Decreases competence because timeout exceeded")
                                self._waiting = False
                                self._phase = Phase.FIND_NEXT_GOAL
                                self.idle_since = None
                                self._remove = False
                                self._to_search.append(self._door['room_name'])
                                self._answered = True
                            else:
                                return None, {}

                    if 'class_inheritance' in info and 'ObstacleObject' in info['class_inheritance'] and 'tree' in info[
                        'obj_id']:
                        objects.append(info)
                        # Communicate which obstacle is blocking the entrance
                        ask : bool = self._decide_to_ask_or_not(willingness, competence)
                        if  self._answered == False and not self._remove and not self._waiting:
                            self._send_message('Found tree blocking  ' + str(self._door['room_name']) + '. Please decide whether to "Remove" or "Continue" searching. \n \n \
                                Important features to consider are: \n safe - victims rescued: ' + str(
                                self._collected_victims) + '\n explore - areas searched: area ' + str(
                                self._searched_rooms).replace('area ', '') + ' \
                                \n clock - removal time: 10 seconds', 'RescueBot')
                            self._obstacle_is_tree = True
                            self._waiting = True
                        # Determine the next area to explore if the human tells the agent not to remove the obstacle
                        if self.received_messages_content and self.received_messages_content[
                            -1] == 'Continue' and not self._remove:
                            self._answered = True
                            self._waiting = False
                            # Add area to the to do list
                            self._to_search.append(self._door['room_name'])
                            self._phase = Phase.FIND_NEXT_GOAL
                            self.idle_since = None
                            self._obstacle_is_tree = False
                        # If decide not to ask, remove it automatically
                        # Remove the obstacle if the human tells the agent to do so
                        if not ask or self.received_messages_content and self.received_messages_content[
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
                            self._obstacle_is_tree = False
                            self.idle_since = None
                            return RemoveObject.__name__, {'object_id': info['obj_id']}
                        # Remain idle untill the human communicates what to do with the tree obstacle
                        else:
                            return None, {}

                    if 'class_inheritance' in info and 'ObstacleObject' in info['class_inheritance'] and 'stone' in \
                            info['obj_id']:
                        objects.append(info)
                        # Communicate which obstacle is blocking the entrance
                        ask: bool = self._decide_to_ask_or_not(willingness, competence)
                        if self._answered == False and not self._remove and not self._waiting:
                            self._send_message('Found stones blocking  ' + str(self._door['room_name']) + '. Please decide whether to "Remove together", "Remove alone", or "Continue" searching. \n \n \
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
                            self.idle_since = None
                            # Add area to the to do list
                            self._to_search.append(self._door['room_name'])
                            self._phase = Phase.FIND_NEXT_GOAL
                        # Remove the obstacle alone if the human decides so or if human is incompetent
                        if not ask or self.received_messages_content and self.received_messages_content[
                            -1] == 'Remove alone' and not self._remove:
                            self._answered = True
                            self._waiting = False
                            self._send_message('Removing stones blocking ' + str(self._door['room_name']) + '.',
                                               'RescueBot')
                            self._phase = Phase.ENTER_ROOM
                            self._remove = False
                            self.idle_since = None
                            return RemoveObject.__name__, {'object_id': info['obj_id']}
                        # Remove the obstacle together if the human decides so
                        if self.received_messages_content and self.received_messages_content[
                            -1] == 'Remove together' or self._remove:
                            if not self._remove:
                                self._answered = True
                            # Tell the human to come over and be idle until human arrives
                            if not state[{'is_human_agent': True}]:
                                self._send_message(
                                    'Please come to ' + str(self._door['room_name']) + ' to remove stones together.',
                                    'RescueBot')

                                if self.idle_since is None:
                                    self.idle_since = self._tick
                                return None, {}
                            # Tell the human to remove the obstacle when he/she arrives
                            if state[{'is_human_agent': True}]:
                                self._send_message('Lets remove stones blocking ' + str(self._door['room_name']) + '!',
                                                   'RescueBot')
                                self.idle_since = None
                                return None, {}
                        # Remain idle until the human communicates what to do with the identified obstacle
                        else:
                            if self.idle_since is not None and self._tick - self.idle_since > self._calculate_timeout(
                                    self._loadBelief(self._team_members, self._folder, baseline), 30):
                                print("UPDATE: Decreases competence because timeout exceeded")
                                competence -= self._calculate_competence_update(trustBeliefs, 0.2 if baseline is None else 0.0)
                                trustBeliefs[self._human_name]['competence'] = competence
                                self._update_csv(competence, willingness)
                                self.idle_since = None
                                self._answered = True
                                self._waiting = False
                                self._send_message('Removing stones blocking ' + str(
                                    self._door['room_name']) + ' because you took too long to come.',
                                                   'RescueBot')
                                self._phase = Phase.ENTER_ROOM
                                self._remove = False
                                self.idle_since = None
                                return RemoveObject.__name__, {'object_id': info['obj_id']}
                            else:
                                return None, {}
                # If no obstacles are blocking the entrance, enter the area
                if len(objects) == 0:
                    self._answered = False
                    self._remove = False
                    self._waiting = False
                    self.idle_since = None
                    self._aid_remove = False
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
                                        self._searched_rooms.append(self._door['room_name'])
                                    # Do not continue searching the rest of the area but start planning to rescue the victim
                                    self._phase = Phase.FIND_NEXT_GOAL

                            # Identify injured victim in the area
                            if 'healthy' not in vic and vic not in self._found_victims:
                                self._recent_vic = vic
                                # Add the victim and the location to the corresponding dictionary
                                self._found_victims.append(vic)
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
                                    if vic in self._collected_victims:
                                        print("You lied in collecting a victim so willingness goes down")
                                        willingness -= 0.5
                                    self._waiting = True

                                if 'critical' in vic and self._answered == False and not self._waiting:
                                    self._send_message('Found ' + vic + ' in ' + self._door['room_name'] + '. Please decide whether to "Rescue" or "Continue" searching. \n\n \
                                        Important features to consider are: \n explore - areas searched: area ' + str(
                                        self._searched_rooms).replace('area',
                                                                      '') + ' \n safe - victims rescued: ' + str(
                                        self._collected_victims) + '\n \
                                        afstand - distance between us: ' + self._distance_human, 'RescueBot')
                                    self._waiting = True
                                    if vic in self._collected_victims:
                                        print("You lied in collecting a victim so willingness goes down")
                                        willingness -= 0.5
                                    # Execute move actions to explore the area
                    return action, {}

                # Communicate that the agent did not find the target victim in the area while the human previously communicated the victim was located here
                if self._goal_vic in self._found_victims and self._goal_vic not in self._room_vics and \
                        self._found_victim_logs[self._goal_vic]['room'] == self._door['room_name']:
                    self._send_message(self._goal_vic + ' not present in ' + str(self._door[
                                                                                     'room_name']) + ' because I searched the whole area without finding ' + self._goal_vic + '.',
                                       'RescueBot')
                    willingness -= self._calculate_willingness_update(trustBeliefs, 0.15 if baseline is None else 0.0)
                    trustBeliefs[self._human_name]['willingness'] = willingness
                    self._update_csv(competence, willingness)
                    # Remove the victim location from memory
                    self._found_victim_logs.pop(self._goal_vic, None)
                    self._found_victims.remove(self._goal_vic)
                    self._room_vics = []
                    # Reset received messages (bug fix)
                    self.received_messages = []
                    self.received_messages_content = []
                # Add the area to the list of searched areas
                if self._door['room_name'] not in self._searched_rooms:
                    self._searched_rooms.append(self._door['room_name'])
                # Make a plan to rescue a found critically injured victim if the human decides so
                if self.received_messages_content and self.received_messages_content[
                    -1] == 'Rescue' and 'critical' in self._recent_vic:
                    self._rescue = 'together'
                    self._answered = True
                    self._waiting = False

                    print("UPDATE: Willingness and Competence increase because a critical victim is rescued together")
                    # Tell the human to come over and help carry the critically injured victim
                    if not state[{'is_human_agent': True}]:
                        self._send_message('Please come to ' + str(self._door['room_name']) + ' to carry ' + str(
                            self._recent_vic) + ' together.', 'RescueBot')
                    # Tell the human to carry the critically injured victim together
                    if state[{'is_human_agent': True}]:
                        self._send_message('Lets carry ' + str(
                            self._recent_vic) + ' together! Please wait until I moved on top of ' + str(
                            self._recent_vic) + '.', 'RescueBot')
                    self._goal_vic = self._recent_vic
                    self._recent_vic = None
                    self._phase = Phase.PLAN_PATH_TO_VICTIM
                # Make a plan to rescue a found mildly injured victim together if the human decides so
                if self.received_messages_content and self.received_messages_content[
                    -1] == 'Rescue together' and 'mild' in self._recent_vic:
                    self._rescue = 'together'
                    self._answered = True
                    self._waiting = False
                    willingness += self._calculate_willingness_update(trustBeliefs, 0.05)
                    competence += self._calculate_competence_update(trustBeliefs, 0.05)

                    trustBeliefs[self._human_name]["willingness"] = willingness
                    trustBeliefs[self._human_name]['competence'] = competence
                    self._update_csv(competence, willingness)

                    print("UPDATE: Willingness and Competence increase because a mildly injured victim is rescued together")
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
                    self._phase = Phase.PLAN_PATH_TO_VICTIM
                # Make a plan to rescue the mildly injured victim alone if the human decides so or if the human is incompetent and or unwilling, and communicate this to the human
                if self.received_messages_content and self.received_messages_content[ # TODO  self._decide_to_ask_or_not(willingness, competence) or
                            -1] == 'Rescue alone' and 'mild' in self._recent_vic:
                    self._send_message('Picking up ' + self._recent_vic + ' in ' + self._door['room_name'] + '.',
                                       'RescueBot')
                    self._rescue = 'alone'
                    self._answered = True
                    self._waiting = False
                    self._goal_vic = self._recent_vic
                    self._goal_loc = self._remaining[self._goal_vic]
                    self._recent_vic = None
                    self._phase = Phase.PLAN_PATH_TO_VICTIM
                # Continue searching other areas if the human decides so
                if self.received_messages_content and self.received_messages_content[-1] == 'Continue':
                    self._answered = True
                    self._waiting = False
                    self._todo.append(self._recent_vic)
                    self._recent_vic = None
                    self._phase = Phase.FIND_NEXT_GOAL
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

            if Phase.FOLLOW_PATH_TO_VICTIM == self._phase:  # Independent of trust
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

            if Phase.TAKE_VICTIM == self._phase:  # Independent of trust
                # Store all area tiles in a list
                room_tiles = [info['location'] for info in state.values()
                              if 'class_inheritance' in info
                              and 'AreaTile' in info['class_inheritance']
                              and 'room_name' in info
                              and info['room_name'] == self._found_victim_logs[self._goal_vic]['room']]
                self._roomtiles = room_tiles
                objects = []
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
                        if not self._human_name in info['name']:
                            self._waiting = True
                            self._moving = False
                            if self.idle_since is not None and self._tick - self.idle_since > self._calculate_timeout(
                                    self._loadBelief(self._team_members, self._folder, baseline), 0):
                                competence -= self._calculate_competence_update(trustBeliefs, 0.2 if baseline is None else 0.0) # Reduce competence since the timeout was exceeded
                                self._update_csv(competence, willingness)
                                print("UPDATE: Decreases competence because timeout exceeded")
                                self._waiting = False
                                self._moving = True
                                self.idle_since = None
                                if 'mildly' in self._goal_vic:
                                    self._rescue = 'alone'
                                    self._goal_loc = self._remaining[self._goal_vic]
                                    self._send_message('Timeout exceeded, I will carry ' + self._goal_vic + 'the victim myself', 'RescueBot')
                                    break
                                else:
                                    self._answered = True
                                    self._todo.append(self._goal_vic)
                                    self._recent_vic = None
                                    self._rescue = None
                                    self._phase = Phase.FIND_NEXT_GOAL
                                    self._goal_vic = None
                                    self._send_message('Timeout exceeded, I will continue searching', 'RescueBot')
                                    return Idle.__name__, {'duration_in_ticks': 25}
                            else:
                                if self.idle_since is None:
                                    self.idle_since = self._tick
                                return None, {}
                # Add the victim to the list of rescued victims when it has been picked up
                if len(objects) == 0 and 'critical' in self._goal_vic or len(
                        objects) == 0 and 'mild' in self._goal_vic and self._rescue == 'together':
                    self.idle_since = None
                    self._waiting = False
                    if self._goal_vic not in self._collected_victims:
                        self._collected_victims.append(self._goal_vic)
                    self._carrying_together = True
                    # Determine the next victim to rescue or search
                    self._phase = Phase.FIND_NEXT_GOAL
                # When rescuing mildly injured victims alone, pick the victim up and plan the path to the drop zone
                if 'mild' in self._goal_vic and self._rescue == 'alone':
                    self._phase = Phase.PLAN_PATH_TO_DROPPOINT
                    if self._goal_vic not in self._collected_victims:
                        self._collected_victims.append(self._goal_vic)
                    self._carrying = True
                    return CarryObject.__name__, {'object_id': self._found_victim_logs[self._goal_vic]['obj_id'],
                                                  'human_name': self._human_name}

            if Phase.PLAN_PATH_TO_DROPPOINT == self._phase:  # Communication: Indepedent of trust
                self._navigator.reset_full()
                # Plan the path to the drop zone
                self._navigator.add_waypoints([self._goal_loc])
                # Follow the path to the drop zone
                self._phase = Phase.FOLLOW_PATH_TO_DROPPOINT

            if Phase.FOLLOW_PATH_TO_DROPPOINT == self._phase:  # Communication: Indepedent of trust
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

            if Phase.DROP_VICTIM == self._phase:  # Communication: Independent of trust
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

            # Update values of competence and willingness if they have been changed during the play
            update = True if willingness != trustBeliefs[self._human_name]["willingness"] or competence != trustBeliefs[self._human_name]['competence'] else False

            # Copying values to disk
            if update:
                print(f"Update, {willingness}, {competence}")
                trustBeliefs[self._human_name]["willingness"] = willingness
                trustBeliefs[self._human_name]['competence'] = competence
                with open(self._folder + '/beliefs/currentTrustBelief.csv', mode='w') as csv_file:
                    csv_writer = csv.writer(csv_file, delimiter=';', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                    csv_writer.writerow(['name', 'competence', 'willingness'])
                    csv_writer.writerow([self._human_name, trustBeliefs[self._human_name]['competence'],
                                         trustBeliefs[self._human_name]['willingness']])

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
        trustBeliefs = self._loadBelief(self._team_members, self._folder, baseline)

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
                    if area not in self._searched_rooms:
                        self._searched_rooms.append(area)
                # If a received message involves team members finding victims, add these victims and their locations to memory
                if msg.startswith("Found:"):
                    # Identify which victim and area it concerns
                    if len(msg.split()) == 6:
                        foundVic = ' '.join(msg.split()[1:4])
                    else:
                        foundVic = ' '.join(msg.split()[1:5])
                    loc = 'area ' + msg.split()[-1]
                    # Add the area to the memory of searched areas
                    if loc not in self._searched_rooms:
                        self._searched_rooms.append(loc)
                    # Add the victim and its location to memory
                    if foundVic not in self._found_victims:
                        self._found_victims.append(foundVic)
                        self._found_victim_logs[foundVic] = {'room': loc}
                    if foundVic in self._found_victims and self._found_victim_logs[foundVic]['room'] != loc:
                        self._found_victim_logs[foundVic] = {'room': loc}
                    # Decide to help the human carry a found victim when the human's condition is 'weak'
                    if condition == 'weak':
                        self._rescue = 'together'
                    # Add the found victim to the to do list when the human's condition is not 'weak'
                    if 'mild' in foundVic and condition != 'weak':
                        self._todo.append(foundVic)
                # If a received message involves team members rescuing victims, add these victims and their locations to memory
                if msg.startswith('Collect:'): # TODO: Apply same as in remove
                    # Identify which victim and area it concerns
                    if len(msg.split()) == 6:
                        collectVic = ' '.join(msg.split()[1:4])
                    else:
                        collectVic = ' '.join(msg.split()[1:5])
                    loc = 'area ' + msg.split()[-1]
                    # Add the area to the memory of searched areas
                    if loc not in self._searched_rooms:
                        self._searched_rooms.append(loc)
                    # Add the victim and location to the memory of found victims
                    # if collectVic not in self._found_victims:
                        # self._found_victims.append(collectVic)
                        # self._found_victim_logs[collectVic] = {'room': loc}
                    if collectVic in self._found_victims and self._found_victim_logs[collectVic]['room'] != loc:
                        self._found_victim_logs[collectVic] = {'room': loc}
                    # Add the victim to the memory of rescued victims when the   human's condition is not weak
                    if condition != 'weak' and collectVic not in self._collected_victims:
                        self._collected_victims.append(collectVic)
                    # Decide to help the human carry the victim together when the human's condition is weak
                    if condition == 'weak':
                        self._rescue = 'together'
                # If a received message involves team members asking for help with removing obstacles, add their location to memory and come over
                if msg.startswith('Remove:'):
                    # Come over immediately when the agent is not carrying a victim
                    competence = trustBeliefs[self._human_name]['competence']
                    we_trust = True if competence > 0.20 else np.random.uniform(0, 1) > 0.6
                    if not self._carrying and we_trust:
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
                    elif self._carrying: # TODO: Check this case
                        area = 'area ' + msg.split()[-1]
                        self._send_message('Will come to ' + area + ' after dropping ' + self._goal_vic + '.', 'RescueBot')
            # Store the current location of the human in memory
            if mssgs and mssgs[-1].split()[-1] in ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13',
                                                   '14']:
                self._human_loc = int(mssgs[-1].split()[-1])

    def _loadBelief(self, members, folder, baseline):
        '''
        Loads trust belief values if agent already collaborated with human before, otherwise trust belief values are initialized using default values.
        '''
        # Create a dictionary with trust values for all team members
        trustBeliefs = {}
        if baseline is None:
            with open(folder + '/beliefs/currentTrustBelief.csv') as csvfile:
                reader = csv.reader(csvfile, delimiter=';', quotechar="'")
                for row in reader:
                    if row and row[0] == self._human_name:
                        trustBeliefs[row[0]] = {'competence': float(row[1]), 'willingness': float(row[2])}
                        return trustBeliefs

        elif baseline == "NEVER-TRUST":
            trustBeliefs[self._human_name] = {'competence': float(-1), 'willingness': float(-1)}
            return trustBeliefs

        elif baseline == "ALWAYS-TRUST":
            trustBeliefs[self._human_name] = {'competence': float(1), 'willingness': float(1)}
            return trustBeliefs

        elif baseline == "RANDOM-TRUST":
            trustBeliefs[self._human_name] = {'competence': random_competence, 'willingness': random_willingness}
            return trustBeliefs

        # Set a default starting trust value
        default = 0.0
        trustfile_header = []
        trustfile_contents = []
        # Check if agent already collaborated with this human before, if yes: load the corresponding trust values, if no: initialize using default trust values
        with open(folder + '/beliefs/allTrustBeliefs.csv') as csvfile:
            reader = csv.reader(csvfile, delimiter=';', quotechar="'")
            for row in reader:
                if trustfile_header == []:
                    trustfile_header = row
                    continue
                # Retrieve trust values 
                if row and row[0] == self._human_name:
                    name = row[0]
                    competence = float(row[1])
                    willingness = float(row[2])
                    trustBeliefs[name] = {'competence': competence, 'willingness': willingness}
                # Initialize default trust values
                if row and row[0] != self._human_name:
                    competence = default
                    willingness = default
                    trustBeliefs[self._human_name] = {'competence': competence, 'willingness': willingness}
        return trustBeliefs

    def _trustBelief(self, tick, members, trustBeliefs, folder, receivedMessages,
                     state, baseline):
        '''
        Baseline implementation of a trust belief. Creates a dictionary with trust belief scores for each team member, for example based on the received messages.
        '''
        agent_beliefs = trustBeliefs[self._human_name]
        all_rooms = state.get_all_room_names().remove('world_bounds')

        for message in receivedMessages:
            print(message)
            # Increase agent trust in a team member that rescued a victim
            action_type = message.split(":")[0]

            if action_type in self._atomic_actions:
                area = message[-1]

                self._objectiveHistory.setdefault(action_type, []).append(
                    Objective(action=action_type, start_time=tick, area=area))

                # Log search goal
                if action_type == 'Search':
                    agent_beliefs['willingness'] += self._calculate_willingness_update(trustBeliefs, 0.05 if baseline is None else 0.0)
                    print("UPDATE: Willingness increased since human said he is going to search area ", area)

                # Log found event
                if action_type == 'Found':
                    if any(obj.area == area for obj in self._objectiveHistory.get('Search', [])):
                        agent_beliefs['willingness'] += self._calculate_willingness_update(trustBeliefs, 0.02 if baseline is None else 0.0)
                        print("UPDATE: Competence increased since human said found a victim in an area he said he would search")
                    else:
                        print("UPDATE: Competence decreased since human said found in an area he wasn't going to search")
                        agent_beliefs['willingness'] -= self._calculate_willingness_update(trustBeliefs, 0.08 if baseline is None else 0.0)

                # Log collect goal
                if action_type == 'Collect':
                    # if area in self._objectiveHistory.get('Search', []):
                    #     agent_beliefs['competence'] += 0.1 if baseline is None else 0.0
                    # else:
                    #     agent_beliefs['competence'] -= 0.02 if baseline is None else 0.0
                    # if area in self._objectiveHistory.get('Found', []):
                    #     print("Competence increased since human said collect a victim in an area he said he found")
                    #     agent_beliefs['competence'] += 0.1 if baseline is None else 0.0
                    # else:
                    #     print("Competence decreased since human said collect a victim he did not say he found")
                    #     agent_beliefs['competence'] -= 0.1 if baseline is None else 0.0

                    if not any(area == obj.area for obj in self._objectiveHistory.get('Search', [])):
                        print("UPDATE: Willingness decreased since it is collecting a victim in an area he was not going to search")
                        agent_beliefs['willingness'] -= self._calculate_willingness_update(trustBeliefs, (0.05 if baseline is None else 0.0))

                    if not any(area == obj.area for obj in self._objectiveHistory.get('Found', [])):
                        print("UPDATE: Willingness decreased since collecting victim he did not found")
                        agent_beliefs['willingness'] -= self._calculate_willingness_update(trustBeliefs, 0.05 if baseline is None else 0.0)

                    self._objectiveHistory.get(action_type, []).append(
                        Objective(action="Rescue together", start_time=tick, area=area))

                if not self._obstacle_is_tree and message == 'Remove':
                    self._aid_remove = True
                    # agent_beliefs['willingness'] += 0.2 if baseline is None else 0.0
                    self._objectiveHistory.setdefault(message, []).append(
                        Objective(action=message, start_time=tick, area=self._human_loc))

            if message == 'Rescue together' or message == 'Rescue':  # Start time for joint rescue
                # agent_beliefs['willingness'] += 0.05 if baseline is None else 0.0 # Increase willingness
                self._objectiveHistory.setdefault('Rescue', []).append(
                    Objective(action=message, start_time=tick, area=self._agent_loc, person=self._recent_vic))

            # Log alone goal: Telling the robot to complete a task alone when it could be done jointly
            # if 'alone' in action_type:
            #     print("Decrease willingness because alone")
            #     agent_beliefs['willingness'] -= 0.1 if baseline is None else 0.0

            # Log message to ask for help when removing
            if action_type == 'Help remove':
                print("UPDATE: Increase willingness because human is willing to collaborate")
                agent_beliefs['willingness'] += self._calculate_willingness_update(trustBeliefs, (0.02 if baseline is None else 0.0))

            # Decrease willingness slightly when asking to continue
            if action_type == 'Continue':
                print("UPDATE: Decrease willingness because human told agent to delay task")
                agent_beliefs['willingness'] -= self._calculate_willingness_update(trustBeliefs, (0.04 if baseline is None else 0.0))

        # Joint Removal event asked from the Robot's side
        if not self._aid_remove:
            for objective in self._objectiveHistory.get('Remove', []):
                if objective.end_time is None:
                    objective.end_time = tick
                    threshold = self._calculate_threshold(agent_beliefs, 'remove', True)
                    if objective.end_time - objective.start_time < threshold:
                        print("UPDATE: Increase competence since it removes within threshold")
                        agent_beliefs['competence'] += self._calculate_competence_update(trustBeliefs, 0.05 if baseline is None else 0.0)
                    else:
                        print("UPDATE: Decreases competence since it removes out of threshold")
                        agent_beliefs['competence'] -= self._calculate_competence_update(trustBeliefs, 0.075 if baseline is None else 0.0)

        # Joint Rescue event asked from the Robot's side
        if self._carrying_together:
            for objective in self._objectiveHistory.get('Rescue', []):
                if objective.end_time is None:
                    objective.end_time = tick
                    if tick - objective.end_time < self._calculate_threshold(agent_beliefs, 'rescue'):
                        if baseline is None:
                            agent_beliefs['competence'] += self._calculate_competence_update(trustBeliefs, 0.05)
                            agent_beliefs['willingness'] += self._calculate_willingness_update(trustBeliefs,  0.05 if (self._goal_vic is not None and "mild" in self._goal_vic) else 0.025)
                            print("UPDATE: Increase both since it rescues within threshold")
                    else:
                        if baseline is None:
                            print("UPDATE: Decreases competence since it rescues out of threshold")
                            agent_beliefs['competence'] -= self._calculate_competence_update(trustBeliefs, (0.1 if (
                                    self._goal_vic is not None and "critical" in self._goal_vic) else 0.05))


        # If all rooms have been searched but not all victims rescued -> human lies -> willingness goes down
        if self._searched_rooms == all_rooms and len(self._found_victims) < 8:
            print("UPDATE: All rooms have been searched but not all victims rescued -> human lies -> willingness goes down")


            trustBeliefs[self._human_name]['willingness'] -= self._calculate_willingness_update(trustBeliefs, (0.3 if baseline is None else 0.0))

        # Restrict the competence and willingness belief to a range of -1 to 1
        trustBeliefs[self._human_name]['willingness'] = np.clip(trustBeliefs[self._human_name]['willingness'], -1, 1)
        trustBeliefs[self._human_name]['competence'] = np.clip(trustBeliefs[self._human_name]['competence'], -1, 1)


        # Save current trust belief values so we can later use and retrieve them to add to a csv file with all the logged trust belief values
        with open(folder + '/beliefs/currentTrustBelief.csv', mode='w') as csv_file:
            csv_writer = csv.writer(csv_file, delimiter=';', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            csv_writer.writerow(['name', 'competence', 'willingness'])
            csv_writer.writerow([self._human_name, trustBeliefs[self._human_name]['competence'],
                                 trustBeliefs[self._human_name]['willingness']])

        self._update_csv(trustBeliefs[self._human_name]['competence'], trustBeliefs[self._human_name]['willingness'])
        trustBeliefs[self._human_name] = agent_beliefs

        print("Tick: " + str(tick) + " " + str(agent_beliefs))
        self._dictionary_to_print[tick] = agent_beliefs
        self._plot_ticks()
        return trustBeliefs

    def _calculate_threshold(self, beliefs: dict[str, int], action: str, distance: bool = False):
        """
        Calculates the dynamic threshold to complete an action

        :param: beliefs: dictionary with willingness and competence values
        """
        threshold = 60

        if distance:
            threshold += self.distances[self._distance_human]

        if action == 'rescue':
            if beliefs['competence'] > 0.1:
                threshold += 5

        if action == 'remove':
            if beliefs['competence'] > 0.1:
                threshold += 10

        return threshold

    def _calculate_timeout(self, trust: dict, rock_offset : int, distance: bool = True):
        """
        Calculates timeout before robot continues with next task

        :param trust: dictionary with willingness and competence values
        :param rock_offset: add extra threshold time depending on the rock type
        :param distance: if set to True it takes into account distance between agent and human
        :returns timeout in ticks
        """
        # Negative willingness reduces timeout
        # Positive willignenss is rewarded with a larger timeout
        to = 100 + 30 * trust.get(self._human_name).get('willingness')

        if distance:
            # Larger distances contribute more time to timeout
            to += self.distances[self._distance_human]

        # Add offset depending whether it is a rock or stone
        to += rock_offset

        return to

    def _decide_to_ask_or_not(self, willingness : float, competence : float) -> bool:
        """
        Decides whether to ask for help when removing a rock or to do it alone

        :param willingness: willingness value
        :param competence: competence value

        :returns whether to ask for help or not
        """
        low_willingness : bool = willingness < -0.5
        high_competence : bool = competence > 0.6

        if low_willingness and high_competence:
            if abs(willingness) > abs(competence):
                # If human is more unwilling than competent, do not ask
                return False
            else:
                # If human is more competent than unwilling (or the same), ask
                return True
        elif low_willingness:
            return False # Low willingness hence agent should not even bother to ask
        else:
            return True # High competence hence agent should ask for help



    def _send_message(self, mssg, sender):
        '''
        send messages from agent to other team members
        '''
        msg = Message(content=mssg, from_id=sender)
        if msg.content not in self.received_messages_content and 'Our score is' not in msg.content:
            self.send_message(msg)
            self._send_messages.append(msg.content)
        # Sending the hidden score message (DO NOT REMOVE)
        if 'Our score is' in msg.content:
            self.send_message(msg)

    def _getClosestRoom(self, state, objs, currentDoor):
        '''
        calculate which area is closest to the agent's location
        '''
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

    def _calculate_competence_update(self, trustBeliefs: dict[str, dict[str, float]], update: float):
        """
        Calculates the update size of the competence based on its current value.
        Bigger values lead to smaller updates.
        See _calculate_update for more information.
        """
        return self._calculate_update(trustBeliefs, update, 'competence')

    def _calculate_willingness_update(self, trustBeliefs: dict[str, dict[str, float]], update: float):
        """
        Calculates the update size of the willingness based on its current value.
        Bigger values lead to smaller updates.
        See _calculate_update for more information.
        """
        return self._calculate_update(trustBeliefs, update, 'willingness')

    def _calculate_update(self, trustBeliefs: dict[str, dict[str, float]], update: float, belief: str):
        """"
        Calculates the update size of the willingness and competence based on their current values.
        Bigger values lead to smaller updates.
        """
        alpha = 0.4
        discount = 1 - (alpha * (trustBeliefs[self._human_name][belief] ** 2))
        return max(min(discount * update, 1), -1)

    def _update_csv(self, competence : float, willingness : float):
        """
        Writes to csv file
        """
        with open(self._folder + '/beliefs/currentTrustBelief.csv', mode='w') as csv_file:
            csv_writer = csv.writer(csv_file, delimiter=';', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            csv_writer.writerow(['name', 'competence', 'willingness'])
            csv_writer.writerow([self._human_name, competence, willingness])

    def _plot_ticks(self, save_path="trust_logs/trust_beliefs_per_tick.csv"):
        """
        Updates competence and willingness values
        """
        os.makedirs(os.path.dirname(save_path), exist_ok=True)  # Creates trust_logs if it doesn't exist
        if save_path:
            with open(save_path, mode='w', newline='') as csv_file:
                csv_writer = csv.writer(csv_file, delimiter=';', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                csv_writer.writerow(['Tick', 'Willingness', 'Competence'])

                ticks = self._dictionary_to_print.keys()
                for tick in ticks:
                    csv_writer.writerow([
                        tick,
                        self._dictionary_to_print[tick]['willingness'],
                        self._dictionary_to_print[tick]['competence']
                    ])
        else:
            ticks = self._dictionary_to_print.keys()
            for tick in ticks:
                print(
                    f"{tick},{self._dictionary_to_print[tick]['willingness']},{self._dictionary_to_print[tick]['competence']}")

