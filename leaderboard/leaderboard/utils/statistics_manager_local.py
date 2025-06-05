#!/usr/bin/env python

# Copyright (c) 2018-2019 Intel Corporation
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
This module contains a statistics manager for the CARLA AD leaderboard
"""

from __future__ import print_function

from dictor import dictor
import math
import sys

from srunner.scenariomanager.traffic_events import TrafficEventType

from leaderboard.utils.checkpoint_tools import fetch_dict, save_dict, create_default_json_msg
import os
import json

PENALTY_COLLISION_PEDESTRIAN = 0.50
PENALTY_COLLISION_VEHICLE = 0.60
PENALTY_COLLISION_STATIC = 0.65
PENALTY_TRAFFIC_LIGHT = 0.70
PENALTY_STOP = 0.80
if os.getenv('BENCHMARK') is not None:
    if os.getenv('BENCHMARK') == 'longest6':
        PENALTY_STOP = 1.00


class RouteRecord():
    def __init__(self):
        self.route_id = None
        self.index = None
        self.timestamp = None
        self.status = 'Started'
        self.infractions = {
            'collisions_pedestrian': [],
            'collisions_vehicle': [],
            'collisions_layout': [],
            'red_light': [],
            'stop_infraction': [],
            'outside_route_lanes': [],
            'route_dev': [],
            'route_timeout': [],
            'vehicle_blocked': []
        }

        self.outside_route_lanes_distance = [] 

        self.scores = {
            'score_route': 0,
            'score_penalty': 0,
            'score_composed': 0
        }

        self.meta = {}


def to_route_record(record_dict):
    record = RouteRecord()
    for key, value in record_dict.items():
        setattr(record, key, value)

    return record


def compute_route_length(config):
    trajectory = config.trajectory

    route_length = 0.0
    previous_location = None
    for location in trajectory:
        if previous_location:
            dist = math.sqrt((location.x-previous_location.x)*(location.x-previous_location.x) +
                             (location.y-previous_location.y)*(location.y-previous_location.y) +
                             (location.z - previous_location.z) * (location.z - previous_location.z))
            route_length += dist
        previous_location = location

    return route_length


class StatisticsManager(object):

    """
    This is the statistics manager for the CARLA leaderboard.
    It gathers data at runtime via the scenario evaluation criteria.
    """

    def __init__(self):
        self._master_scenario = None
        self._registry_route_records = []

    def resume(self, endpoint):
        data = fetch_dict(endpoint)

        if data and dictor(data, '_checkpoint.records'):
            records = data['_checkpoint']['records']

            for record in records:
                self._registry_route_records.append(to_route_record(record))

    def set_route(self, route_id, index):

        self._master_scenario = None
        route_record = RouteRecord()
        route_record.route_id = route_id
        route_record.index = index

        if index < len(self._registry_route_records):
            # the element already exists and therefore we update it
            self._registry_route_records[index] = route_record
        else:
            self._registry_route_records.append(route_record)

    def set_scenario(self, scenario):
        """
        Sets the scenario from which the statistics willb e taken
        """
        self._master_scenario = scenario

    def compute_route_statistics(self, config, route_date_string, duration_time_system=-1, duration_time_game=-1, failure=""):
        """
        Compute the current statistics by evaluating all relevant scenario criteria
        """
        index = config.index

        if not self._registry_route_records or index >= len(self._registry_route_records):
            raise Exception('Critical error with the route registry.')

        # fetch latest record to fill in
        route_record = self._registry_route_records[index]

        target_reached = False
        score_penalty = 1.0
        score_route = 0.0

        route_record.meta['duration_system'] = duration_time_system
        route_record.meta['duration_game'] = duration_time_game
        route_record.meta['route_length'] = compute_route_length(config)
        route_record.timestamp = route_date_string

        if self._master_scenario:
            if self._master_scenario.timeout_node.timeout:
                route_record.infractions['route_timeout'].append('Route timeout.')
                failure = "Agent timed out"

            for node in self._master_scenario.get_criteria():
                if node.list_traffic_events:
                    # analyze all traffic events
                    for event in node.list_traffic_events:
                        if event.get_type() == TrafficEventType.COLLISION_STATIC:
                            score_penalty *= PENALTY_COLLISION_STATIC
                            route_record.infractions['collisions_layout'].append(event.get_message())

                        elif event.get_type() == TrafficEventType.COLLISION_PEDESTRIAN:
                            score_penalty *= PENALTY_COLLISION_PEDESTRIAN
                            route_record.infractions['collisions_pedestrian'].append(event.get_message())

                        elif event.get_type() == TrafficEventType.COLLISION_VEHICLE:
                            score_penalty *= PENALTY_COLLISION_VEHICLE
                            route_record.infractions['collisions_vehicle'].append(event.get_message())

                        elif event.get_type() == TrafficEventType.OUTSIDE_ROUTE_LANES_INFRACTION:
                            score_penalty *= (1 - event.get_dict()['percentage'] / 100)
                            route_record.infractions['outside_route_lanes'].append(event.get_message())
                            route_record.outside_route_lanes_distance.append(event.get_dict()['distance'])

                        elif event.get_type() == TrafficEventType.TRAFFIC_LIGHT_INFRACTION:
                            score_penalty *= PENALTY_TRAFFIC_LIGHT
                            route_record.infractions['red_light'].append(event.get_message())

                        elif event.get_type() == TrafficEventType.ROUTE_DEVIATION:
                            route_record.infractions['route_dev'].append(event.get_message())
                            failure = "Agent deviated from the route"

                        elif event.get_type() == TrafficEventType.STOP_INFRACTION:
                            score_penalty *= PENALTY_STOP
                            route_record.infractions['stop_infraction'].append(event.get_message())

                        elif event.get_type() == TrafficEventType.VEHICLE_BLOCKED:
                            route_record.infractions['vehicle_blocked'].append(event.get_message())
                            failure = "Agent got blocked"

                        elif event.get_type() == TrafficEventType.ROUTE_COMPLETED:
                            score_route = 100.0
                            target_reached = True
                        elif event.get_type() == TrafficEventType.ROUTE_COMPLETION:
                            if not target_reached:
                                if event.get_dict():
                                    score_route = event.get_dict()['route_completed']
                                else:
                                    score_route = 0

        # update route scores
        route_record.scores['score_route'] = score_route
        route_record.scores['score_penalty'] = score_penalty
        route_record.scores['score_composed'] = max(score_route*score_penalty, 0.0)

        # update status
        if target_reached:
            route_record.status = 'Completed'
        else:
            route_record.status = 'Failed'
            if failure:
                route_record.status += ' - ' + failure

        return route_record

    def compute_global_statistics(self, total_routes):
        global_record = RouteRecord()
        global_record.route_id = -1
        global_record.index = -1
        global_record.status = 'Completed'

        total_driven_kms = 0.0
        total_infractions = {key: 0 for key in global_record.infractions.keys()}
        total_outside_route_lanes_distance_m = 0.0

        if self._registry_route_records:
            for route_record in self._registry_route_records:
                global_record.scores['score_route'] += route_record.scores['score_route']
                global_record.scores['score_penalty'] += route_record.scores['score_penalty']
                global_record.scores['score_composed'] += route_record.scores['score_composed']

                route_length_kms = max(route_record.scores['score_route']/100.0 * route_record.meta['route_length'] / 1000.0, 0.001)
                total_driven_kms += route_length_kms

                for key in global_record.infractions.keys():
                    total_infractions[key] += len(route_record.infractions[key])
                    
                total_outside_route_lanes_distance_m += sum(route_record.outside_route_lanes_distance)

                if route_record.status != 'Completed':
                    global_record.status = 'Failed'
                    if 'exceptions' not in global_record.meta:
                        global_record.meta['exceptions'] = []
                    global_record.meta['exceptions'].append((
                        route_record.route_id,
                        route_record.index,
                        route_record.status
                    ))

            # After all routes are processed
            for key in global_record.infractions.keys():
                if total_driven_kms > 0:
                    global_record.infractions[key] = total_infractions[key] / total_driven_kms
                else:
                    global_record.infractions[key] = 0.0
            
            global_record.meta['total_driven_kms'] = total_driven_kms
            global_record.meta['total_outside_route_lanes_distance_m'] = total_outside_route_lanes_distance_m
            global_record.meta['total_outside_route_lanes_distance_per_km'] = total_outside_route_lanes_distance_m / 1000.0 / total_driven_kms if total_driven_kms > 0 else 0.0
            global_record.meta['total_driven_kms_per_route'] = total_driven_kms / float(total_routes) if total_routes > 0 else 0.0
            global_record.meta['total_routes'] = total_routes
            global_record.meta['total_routes_completed'] = len([record for record in self._registry_route_records if record.status == 'Completed'])
            global_record.meta['total_routes_failed'] = len([record for record in self._registry_route_records if record.status != 'Completed'])

            global_record.scores['score_route'] /= float(total_routes)
            global_record.scores['score_penalty'] /= float(total_routes)
            global_record.scores['score_composed'] /= float(total_routes)

        return global_record

    def compute_and_save_route_statistics(self, config, route_date_string, timestamp, filename):
        """
        Compute the route statistics independently and write to a JSON file without modifying any existing records.
        """
        result = {
            "route": route_date_string,
            "status": "Failed",
            "failure_reason": None,
            "timestamp": timestamp,
            "route_length_meters": compute_route_length(config),
            "scores": {
                "route_score": 0.0,
                "penalty_score": 1.0,
                "composed_score": 0.0,
            },
            "infractions": {}
        }

        target_reached = False
        score_penalty = 1.0
        score_route = 0.0
        failure_reason = ""

        if self._master_scenario:
            if self._master_scenario.timeout_node and self._master_scenario.timeout_node.timeout:
                result["infractions"].setdefault('route_timeout', []).append({"message": "Route timeout.", "time": route_date_string})
                failure_reason = "Agent timed out"

            for node in self._master_scenario.get_criteria():
                if node.list_traffic_events:
                    for event in node.list_traffic_events:
                        event_type = event.get_type()
                        event_dict = event.get_dict() or {}

                        event_info = {
                            "message": event.get_message(),
                            "dict:": event_dict,
                        }

                        if event_type == TrafficEventType.COLLISION_STATIC:
                            score_penalty *= PENALTY_COLLISION_STATIC
                            result["infractions"].setdefault('collisions_layout', []).append(event_info)

                        elif event_type == TrafficEventType.COLLISION_PEDESTRIAN:
                            score_penalty *= PENALTY_COLLISION_PEDESTRIAN
                            result["infractions"].setdefault('collisions_pedestrian', []).append(event_info)

                        elif event_type == TrafficEventType.COLLISION_VEHICLE:
                            score_penalty *= PENALTY_COLLISION_VEHICLE
                            result["infractions"].setdefault('collisions_vehicle', []).append(event_info)

                        elif event_type == TrafficEventType.OUTSIDE_ROUTE_LANES_INFRACTION:
                            percentage = event_dict.get('percentage', 0)
                            score_penalty *= (1 - percentage / 100)
                            result["infractions"].setdefault('outside_route_lanes', []).append(event_info)

                        elif event_type == TrafficEventType.TRAFFIC_LIGHT_INFRACTION:
                            score_penalty *= PENALTY_TRAFFIC_LIGHT
                            result["infractions"].setdefault('red_light', []).append(event_info)

                        elif event_type == TrafficEventType.ROUTE_DEVIATION:
                            result["infractions"].setdefault('route_dev', []).append(event_info)
                            failure_reason = "Agent deviated from the route"

                        elif event_type == TrafficEventType.STOP_INFRACTION:
                            score_penalty *= PENALTY_STOP
                            result["infractions"].setdefault('stop_infraction', []).append(event_info)

                        elif event_type == TrafficEventType.VEHICLE_BLOCKED:
                            result["infractions"].setdefault('vehicle_blocked', []).append(event_info)
                            failure_reason = "Agent got blocked"

                        elif event_type == TrafficEventType.ROUTE_COMPLETED:
                            score_route = 100.0
                            target_reached = True

                        elif event_type == TrafficEventType.ROUTE_COMPLETION:
                            if not target_reached:
                                score_route = event_dict.get('route_completed', 0)

        # Fill scores
        result["scores"]["route_score"] = score_route
        result["scores"]["penalty_score"] = score_penalty
        result["scores"]["composed_score"] = max(score_route * score_penalty, 0.0)

        # Determine final status
        if target_reached:
            result["status"] = "Completed"
        else:
            if failure_reason:
                result["failure_reason"] = failure_reason
            else:
                result["failure_reason"] = "Unknown failure"

        # Save JSON file
        with open(filename, 'w') as f:
            json.dump(result, f, indent=4)

    @staticmethod
    def save_record(route_record, index, endpoint):
        data = fetch_dict(endpoint)
        if not data:
            data = create_default_json_msg()

        stats_dict = route_record.__dict__
        record_list = data['_checkpoint']['records']
        if index > len(record_list):
            print('Error! No enough entries in the list')
            sys.exit(-1)
        elif index == len(record_list):
            record_list.append(stats_dict)
        else:
            record_list[index] = stats_dict

        save_dict(endpoint, data)

    @staticmethod
    def save_global_record(route_record, sensors, total_routes, endpoint):
        data = fetch_dict(endpoint)
        if not data:
            data = create_default_json_msg()

        stats_dict = route_record.__dict__
        data['_checkpoint']['global_record'] = stats_dict
        data['values'] = ['{:.3f}'.format(stats_dict['scores']['score_composed']),
                          '{:.3f}'.format(stats_dict['scores']['score_route']),
                          '{:.3f}'.format(stats_dict['scores']['score_penalty']),
                          # infractions
                          '{:.3f}'.format(stats_dict['infractions']['collisions_pedestrian']),
                          '{:.3f}'.format(stats_dict['infractions']['collisions_vehicle']),
                          '{:.3f}'.format(stats_dict['infractions']['collisions_layout']),
                          '{:.3f}'.format(stats_dict['infractions']['red_light']),
                          '{:.3f}'.format(stats_dict['infractions']['stop_infraction']),
                          '{:.3f}'.format(stats_dict['infractions']['outside_route_lanes']),
                          '{:.3f}'.format(stats_dict['infractions']['route_dev']),
                          '{:.3f}'.format(stats_dict['infractions']['route_timeout']),
                          '{:.3f}'.format(stats_dict['infractions']['vehicle_blocked'])
                          ]

        data['labels'] = ['Avg. driving score',
                          'Avg. route completion',
                          'Avg. infraction penalty',
                          'Collisions with pedestrians',
                          'Collisions with vehicles',
                          'Collisions with layout',
                          'Red lights infractions',
                          'Stop sign infractions',
                          'Off-road infractions',
                          'Route deviations',
                          'Route timeouts',
                          'Agent blocked'
                          ]

        entry_status = "Finished"
        eligible = True

        route_records = data["_checkpoint"]["records"]
        progress = data["_checkpoint"]["progress"]

        if progress[1] != total_routes:
            raise Exception('Critical error with the route registry.')

        if len(route_records) != total_routes or progress[0] != progress[1]:
            entry_status = "Finished with missing data"
            eligible = False
        else:
            for route in route_records:
                route_status = route["status"]
                if "Agent" in route_status:
                    entry_status = "Finished with agent errors"
                    break

        data['entry_status'] = entry_status
        data['eligible'] = eligible

        save_dict(endpoint, data)

    @staticmethod
    def save_sensors(sensors, endpoint):
        data = fetch_dict(endpoint)
        if not data:
            data = create_default_json_msg()

        if not data['sensors']:
            data['sensors'] = sensors

            save_dict(endpoint, data)

    @staticmethod
    def save_entry_status(entry_status, eligible, endpoint):
        data = fetch_dict(endpoint)
        if not data:
            data = create_default_json_msg()

        data['entry_status'] = entry_status
        data['eligible'] = eligible
        save_dict(endpoint, data)

    @staticmethod
    def clear_record(endpoint):
        if not endpoint.startswith(('http:', 'https:', 'ftp:')):
            with open(endpoint, 'w') as fd:
                fd.truncate(0)
