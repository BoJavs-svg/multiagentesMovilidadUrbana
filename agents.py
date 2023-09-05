from typing import Any
import matplotlib.pyplot as plt
import random
import heapq
from mesa.model import Model
import numpy as np
import matplotlib.animation as animation
from mesa import Agent, Model
from mesa.time import RandomActivation
from mesa.space import MultiGrid
import math
import json
class World:
    def __init__(self):
        #Generate map
        self.map=np.zeros((13, 13), dtype=int)
        self.entity_map = [[' ' for _ in range(13)] for _ in range(13)]
        drive=[3,4,5,7,8,9]
        for line in drive:
            self.map[line,:]=1
            self.map[:,line]=1
        self.map[6][6]=1
        self.car_spawn_points=[[(12,3),(12,4),(12,5)],
                        [(3,0),(4,0),(5,0)],
                        [(0,7),(0,8),(0,9)],
                        [(7,12),(8,12),(9,12)]               
                        ]
        self.p_spawn_points=[[(12,2),(12,10)],
                            [(2,12),(10,12)],
                            [(0,2),(0,10)],
                            [(2,0),(10,0)]
                        ]
        self.cross_road=[[(10,3),(10,4),(10,5)],
                        [(3,2),(4,2),(5,2)],
                        [(2,7),(2,8),(2,9)],
                        [(7,10),(8,10),(9,10)]]
        self.opposit_cross_road=[[(2,3),(2,4),(2,5)],
                        [(3,10),(4,10),(5,10)],
                        [(10,7),(10,8),(10,9)],
                        [(7,2),(8,2),(9,2)]]
        #Positions
        self.stoplights=[]
        self.car_position = {}
        self.people_position = {}
    def add_person(self,p_id):
        row,col=random.choice(random.choice(self.p_spawn_points))
        if self.entity_map[row][col]!='p':
            self.entity_map[row][col]='p'
            self.people_position[p_id]=(row,col)
            return row,col
        else:
            return (-1,-1)
    def add_car(self,side,car_id):
        i=0
        row,col=self.car_spawn_points[side][i]
        while self.entity_map[row][col]!= ' ' and i <=2:
            row,col=self.car_spawn_points[side][i]
            i+=1
        if not i==4:
            self.entity_map[row][col] = 'c'
            self.car_position[car_id]=(row,col)
            return row,col
        else:
            return (-1,-1)
    def add_stop(self,stop):
        self.stoplights.append(stop)
        
    def move_car(self,car_id,new_position):
        if car_id in self.car_position:
            current_position=self.car_position[car_id]
            self.entity_map[current_position[0]][current_position[1]]=' '
            self.entity_map[new_position[0]][new_position[1]]='c'
            self.car_position[car_id]=new_position
    def move_p(self,p_id,new_position):
        if p_id in self.people_position:
            current_position=self.people_position[p_id]
            self.entity_map[current_position[0]][current_position[1]]=' '
            self.entity_map[new_position[0]][new_position[1]]='p'
            self.people_position[p_id]=new_position
    def get_car_pos(self,car_id):
        return self.car_position[car_id]
    def get_p_pos(self,p_id):
        return self.people_position[p_id]
    def get_car_front(self, car_id, destination):
        current_pos = self.get_car_pos(car_id)

        # Determine the direction the car is going
        if current_pos[0] == destination[0]:
            # Car is moving horizontally (same row)
            direction = 1 if current_pos[1] < destination[1] else -1
        elif current_pos[1] == destination[1]:
            # Car is moving vertically (same column)
            direction = 1 if current_pos[0] < destination[0] else -1
        # Initialize the distance to a large value
        distance = 12
        # Iterate in the direction of movement and find the nearest 'c'
        if current_pos[0] == destination[0]:
            # Moving horizontally
            row = self.entity_map[current_pos[0]]
            col_index = current_pos[1]+direction
            while 0 <= col_index < len(row):
                if row[col_index] != ' ':
                    distance = abs(col_index - current_pos[1])
                    break
                col_index += direction
        else:
            # Moving vertically
            col = [self.entity_map[row][current_pos[1]] for row in range(13)]
            row_index = current_pos[0]+direction
            while 0 <= row_index < len(col):
                if col[row_index] == 'c':
                    distance = abs(row_index - current_pos[0])
                    break
                row_index += direction

        return distance
    def pop_agent(self,id,current_position):
        if id in self.car_position:
            self.entity_map[current_position[0]][current_position[1]]=' '
            self.car_position.pop(id)
        elif id in self.people_position:
            self.entity_map[current_position[0]][current_position[1]]=' '
            self.people_position.pop(id)
    def count_car(self, side):
        def count_cars_in_region(start_row, end_row, start_col, end_col):
            count = 0
            for row in range(start_row, end_row + 1):
                for col in range(start_col, end_col + 1):
                    if self.entity_map[row][col] == 'c':
                        count += 1
            return count

        if side == 0:
            return count_cars_in_region(10, 12, 3, 5)
        elif side == 1:
            return count_cars_in_region(3, 5, 0, 2)
        elif side == 2:
            return count_cars_in_region(0, 2, 7, 9)
        elif side == 3:
            return count_cars_in_region(7, 9, 10, 12)
        else:
            return 0  # Return 0 for invalid side values or handle it as needed
    def clear_crossroad(self,side,set):
        for space in self.cross_road[side]:
            self.map[space[0]][space[1]]=set #0 o 1
        for space in self.opposit_cross_road[side]:
            self.map[space[0]][space[1]]=set #0 o 1
    def display_map(self):
        ax.clear()    
        # Paths
        for row in range(13):
            for col in range(13):
                if self.map[row][col] == 0:
                    ax.add_patch(plt.Rectangle((col, 12 - row), 1, 1, color='green'))  # Walkable area
                elif self.map[row][col] == 1:
                    ax.add_patch(plt.Rectangle((col, 12 - row), 1, 1, color='black'))  # Driveable area
        
        #Entities
        for row in range(13):
            for col in range(13):
                if self.entity_map[row][col] == 'c':
                    ax.plot(col + 0.5, 12 - row + 0.5, 'r^', markersize=10)  # Car
                elif self.entity_map[row][col] == 'p':
                    ax.plot(col + 0.5, 12 - row + 0.5, 'bs', markersize=10)  # Person
        
        legend_handles = [plt.Line2D([0], [0], marker='o', color=stoplight.state, label=f'Stoplight {i+1} - {stoplight.state}', markersize=10, markerfacecolor=stoplight.state) for i, stoplight in enumerate(self.stoplights)]
        ax.legend(handles=legend_handles, loc='upper left')
        
        ax.set_xlim(0, 13)
        ax.set_ylim(0, 13)
        ax.invert_yaxis()
        ax.set_xticks(range(14))  # Adjust the range to include 13
        ax.set_yticks(range(14))  # Adjust the range to include 13
        ax.grid()

class model(Model):
    def __init__(self,n_car,n_people):
        self.m=World()
        self.last_id=0
        self.schedule = RandomActivation(self)
        self.stoplight_agents = []
        self.t_performance = []
        self.c_performance = []
        for s in range(4):
            stoplight = StoplightAgent(s,self.m)
            self.m.add_stop(stoplight)
            self.schedule.add(stoplight)
            self.stoplight_agents.append(stoplight)
            
        side=0
        for i in range(4,n_car+4):
            row,col=self.m.add_car(side,i)
            if (row,col) !=(-1,-1):
                agent = carAgent(i, self.m, row, col, self.stoplight_agents[side], side)
                side = side+1 if side!=3 else 0
                self.schedule.add(agent)
                self.last_id=i
        i=self.last_id
        for _ in range(i+1,n_people+i+1):
            row,col=self.m.add_person(_)
            if (row,col) !=(-1,-1):
                agent=peopleAgent(_,self.m,row,col)
                self.schedule.add(agent)
                self.last_id=_+1
        self.auction_agent = AuctionAgent(unique_id="Auction", model=self, stoplights=self.stoplight_agents)
        self.schedule.add(self.auction_agent)
        self.last_id=_+1
    def step(self):
        agents_to_remove = []
        for agent in self.schedule.agents:
            agent.step()
            if (isinstance(agent, carAgent) or isinstance(agent, peopleAgent)) and agent.beliefs['position'] == agent.desires['destination']:
                agents_to_remove.append(agent)

        for agent in agents_to_remove:
            self.schedule.remove(agent)

        spawn=random.randint(0,1)
        if spawn==1:
            self.spawn_car()
            self.spawn_person()

        t,c=self.calculate_performance()
        self.t_performance.append(t)
        self.c_performance.append(c)
    def calculate_performance(self):
        t = np.mean([agent.time for agent in self.stoplight_agents])
        c = np.mean([agent.num_cars for agent in self.stoplight_agents])
        return t,c
    def spawn_car(self):
            i=random.randint(0,3)
            while True:
                row,col=self.m.add_car(i,self.last_id+1)
                if (row,col) !=(-1,-1):
                    self.last_id+=1
                    agent = carAgent(self.last_id, self.m, row, col, self.stoplight_agents[i],i)
                    self.schedule.add(agent)
                    break
                else:
                    i=random.randint(0,3)
    def spawn_person(self):
        while True:
            row,col=self.m.add_person(self.last_id+1)
            if (row,col) !=(-1,-1):
                self.last_id+=1
                agent=peopleAgent(self.last_id,self.m,row,col)
                self.schedule.add(agent)
                break

class peopleAgent(Agent):
    def __init__(self, unique_id,map,row,col):
        super().__init__(unique_id, model)
        self.m=map
        self.row=row
        self.col=col
        self.beliefs={'position' : self.m.get_p_pos(self.unique_id),
                    'speed' : 1
                    }
        self.desires={'destination': self.calculate_destination()}

        self.intentions = {'move': None}
    def calculate_destination(self):
        current_position = self.m.get_p_pos(self.unique_id)
        d={0:12,12:0}
        if current_position[0]==0 or current_position[0]==12:
            row=d[current_position[0]]
            col=current_position[1]
        else:
            col=d[current_position[1]]
            row=current_position[0]
        return(row,col)
    def calculate_direction(self, pos1, pos2):
        dx = pos2[0] - pos1[0]
        dy = pos2[1] - pos1[1]
        magnitude = math.sqrt(dx**2 + dy**2)
        if magnitude == 0:
            return (0, 0)  # Avoid division by zero
        else:
            return (dx / magnitude, dy / magnitude)
    
    def perceive(self):
        self.beliefs['position']=self.m.get_p_pos(self.unique_id)

    def choose_intention(self):
        self.beliefs['speed']=1
        current_position = self.beliefs['position']
        destination_position = self.desires['destination']
        # Calculate the direction vector
        direction = self.calculate_direction(current_position, destination_position)
        # Calculate the new position
        new_position = (
            int(current_position[0] + direction[0]),
            int(current_position[1] + direction[1])
        )

        if self.m.map[new_position[0]][new_position[1]] == 0:
            self.beliefs['speed']=1
        else:
            if self.m.map[current_position[0]][current_position[1]]==1:
                self.beliefs['speed']=1
            else:
                self.beliefs['speed']=0
        if self.m.entity_map[new_position[0]][new_position[1]] == 'c':
            self.beliefs['speed']=0

        if self.beliefs['speed'] == 1:
            self.intentions['move'] = new_position
    def act(self):
        if self.intentions['move']:
            self.m.move_p(self.unique_id, self.intentions['move'])
        if self.beliefs['position']==self.desires['destination']:
            self.m.pop_agent(self.unique_id,self.beliefs['position'])
    def step(self):
        self.perceive()
        self.choose_intention()
        self.act()

class carAgent(Agent):
    def __init__(self, unique_id, map, row, col, stoplight,lane):
        super().__init__(unique_id, model)
        self.m=map
        self.row=row
        self.col=col
        self.stoplight = stoplight  # List of stoplight agents
        self.lane=lane
        self.beliefs={'position' : self.m.get_car_pos(self.unique_id),
                    'frontDistance' : None,
                    'stopLight' : None,
                    'speed' : 1
                    }
        self.desires={'destination': self.calculate_destination()}
        self.intentions = {'move': None}

    def calculate_destination(self):
        current_position = self.m.get_car_pos(self.unique_id)
        # Calculate destination based on current position and desired direction
        if current_position[0] == 12:  # If the car starts at line 12
            destination = (0, current_position[1])
        elif current_position[0] == 0:  # If the car starts at line 0
            destination = (12, current_position[1])
        elif current_position[1] == 12:  # If the car starts at column 12
            destination = (current_position[0], 0)
        elif current_position[1] == 0:  # If the car starts at column 0
            destination = (current_position[0], 12)
        return destination
    def calculate_direction(self, pos1, pos2):
        dx = pos2[0] - pos1[0]
        dy = pos2[1] - pos1[1]
        magnitude = math.sqrt(dx**2 + dy**2)
        if magnitude == 0:
            return (0, 0)  # Avoid division by zero
        else:
            return (dx / magnitude, dy / magnitude)
    
    def perceive(self):
        self.beliefs['position']=self.m.get_car_pos(self.unique_id)
        self.beliefs['frontDistance']=self.m.get_car_front(self.unique_id, self.desires['destination'])
        self.beliefs['stopLight']=self.stoplight.state


    def choose_intention(self):
        self.beliefs['speed'] = 1
        current_position = self.beliefs['position']
        destination_position = self.desires['destination']

        # Calculate the direction vector
        direction = self.calculate_direction(current_position, destination_position)

        # Calculate the new position
        new_position = (
            int(current_position[0] + direction[0]),
            int(current_position[1] + direction[1])
        )
        if self.beliefs['stopLight'] == 'red' or self.beliefs['stopLight'] == 'yellow':
            if (new_position in self.m.cross_road[self.lane]): 
                self.beliefs['speed'] = 0
            else:
                self.beliefs['speed'] = 1
                
        if self.beliefs['frontDistance'] <= 1:
                self.beliefs['speed'] = 0

        if self.beliefs['speed'] == 1:
            self.intentions['move'] = new_position
    def act(self):
        if self.intentions['move']:
            self.m.move_car(self.unique_id, self.intentions['move'])
        if self.beliefs['position']==self.desires['destination']:
            self.m.pop_agent(self.unique_id,self.beliefs['position'])

    def step(self):
        self.perceive()
        self.choose_intention()
        self.act()

class StoplightAgent(Agent):
    def __init__(self, unique_id,map):
        super().__init__(unique_id, model)
        self.state ="red"  # Initial red
        self.m=map
        self.num_cars=self.m.count_car(self.unique_id)
        self.time=0
        self.bid_price = 0  # Initialize bid price to 0
        self.current_state_duration = 0
        self.max_green_duration = 5 
        self.max_yellow_duration = 3 

    def bid(self):    
        # Calculate bid price based on time waited and number of cars
        time_weight = 0.1  
        cars_weight = 0.9 
        self.bid_price = (time_weight * self.time) + (cars_weight * self.num_cars)

    def step(self):
        # Check the timer to change the state
        self.time+=1
        self.num_cars=self.m.count_car(self.unique_id)
        if self.state == "green":
            self.current_state_duration += 1
            if self.current_state_duration >= self.max_green_duration:
                self.state = "yellow"
                self.current_state_duration = 0
        elif self.state == "yellow":
            self.current_state_duration += 1
            if self.current_state_duration >= self.max_yellow_duration:
                self.state = "red"
                self.current_state_duration = 0
class AuctionAgent(Agent):
    def __init__(self, unique_id, model, stoplights):
        super().__init__(unique_id, model)
        self.stoplights = stoplights
        self.auction_active = True  # Auction is initially active
        self.winning_stoplight = None  # Store the winning stoplight here
        self.time=0 

    def bid(self):
        if self.auction_active:
            # Calculate bid for each stoplight
            for stoplight in self.stoplights:
                # Your bidding logic here; you can consider time waited, number of cars, etc.
                stoplight.bid()
            # Determine the winning stoplight based on bids
            self.winning_stoplight = max(self.stoplights, key=lambda x: x.bid_price)

            # Activate the winning stoplight and deactivate others
            for stoplight in self.stoplights:
                if stoplight == self.winning_stoplight:
                    stoplight.state = "green"
                    stoplight.time = 0  # Reset time waited
                else:
                    stoplight.state = "red"

            # Deactivate the auction
            self.auction_active = False
            self.time=self.winning_stoplight.max_green_duration + self.winning_stoplight.max_yellow_duration

    def step(self):
        # Perform bidding at the beginning of each step
        if self.time==0:
            self.auction_active=True
        else:
            self.time-=1
        self.bid()

def animate(frame, model):
    model.step()  # Step the model
    ax.clear()  # Clear the current plot
    model.m.display_map()  # Display the updated map

n_car=10
n_people=1
t = 1
if t == 0:
    ##Test display crude matplotlib visuals
    mod = model(n_car=n_car,n_people=n_people)
    fig, ax = plt.subplots(figsize=(7, 7))
    ani = animation.FuncAnimation(fig, animate, frames=100, repeat=False, fargs=(mod,))

    plt.show()
    time = mod.t_performance
    cars = mod.c_performance
    fig, ((ax1, ax2),(ax3,ax4)) = plt.subplots(2,2, figsize=(15, 12))
    fig.tight_layout(pad=10.0)
    ax1.hist(time,edgecolor='black')
    ax1.set_xlabel('Time')
    ax1.set_title('Time stopped')

    ax2.hist(cars,edgecolor='black')
    ax2.set_xlabel('Cars')
    ax2.set_title('Cars waiting')
    
    ax3.plot(range(len(time)),time)
    ax3.set_xlabel("Steps")
    ax3.set_ylabel("Time waited")
    ax3.set_title("Time waited per attempt")
    

    ax4.plot(range(len(cars)),cars)
    ax4.set_xlabel("Steps")
    ax4.set_ylabel("Cars waiting")
    ax4.set_title("Cars waiting per step per attempt")
    plt.show()
elif t == 1:
    ##Test to check statistics
    time_hist= []
    cars_hist= []
    time = []
    cars=[]
    num_attempts = 100
    fig, ((ax1, ax2),(ax3,ax4),(ax5,ax6)) = plt.subplots(3,2, figsize=(15, 12))
    fig.tight_layout(pad=10.0)
    for attempt in range(num_attempts):
        print(f"Attempt: {attempt}")
        mod=model(n_car=n_car,n_people=n_people)
        for _ in range(100):
            mod.step()
        time_hist.extend(mod.t_performance)
        cars_hist.extend(mod.c_performance)
        cars.append(mod.c_performance)
        time.append(mod.t_performance)

    ax1.hist(time_hist,edgecolor='black')
    ax1.set_xlabel('Time')
    ax1.set_title('Time stopped')

    ax2.hist(cars_hist,edgecolor='black')
    ax2.set_xlabel('Cars')
    ax2.set_title('Cars waiting')

    for t in time:
        ax3.plot(range(len(t)),t)
    ax3.set_xlabel("Steps")
    ax3.set_ylabel("Time waited")
    ax3.set_title("Time waited per attempt")
    
    for c in cars:
        ax4.plot(range(len(c)),c)
    ax4.set_xlabel("Steps")
    ax4.set_ylabel("Cars waiting")
    ax4.set_title("Cars waiting per step per attempt")
    
    # Sort the data based on performance metric (e.g., minimum and maximum time waited)
    best_case_idx = min(range(len(time)), key=lambda i: max(time[i]))
    worst_case_idx = max(range(len(time)), key=lambda i: max(time[i]))

    best_time = time[best_case_idx]
    worst_time = time[worst_case_idx]

    # Sort the data based on performance metric (e.g., minimum and maximum cars waiting)
    best_case_idx_cars = min(range(len(cars)), key=lambda i: max(cars[i]))
    worst_case_idx_cars = max(range(len(cars)), key=lambda i: max(cars[i]))

    best_cars = cars[best_case_idx_cars]
    worst_cars = cars[worst_case_idx_cars]

    # Plot the best and worst cases in ax3 and ax4
    ax5.plot(range(len(best_time)), best_time, label="Best Case")
    ax5.plot(range(len(worst_time)), worst_time, label="Worst Case")
    ax5.set_xlabel("Steps")
    ax5.set_ylabel("Time waited")
    ax5.set_title("Time waited for Best and Worst Cases")
    ax5.legend()

    ax6.plot(range(len(best_cars)), best_cars, label="Best Case")
    ax6.plot(range(len(worst_cars)), worst_cars, label="Worst Case")
    ax6.set_xlabel("Steps")
    ax6.set_ylabel("Cars waiting")
    ax6.set_title("Cars waiting for Best and Worst Cases")
    ax6.legend()
    
    plt.show()
else:
 # Simulación para json
    mod = model(n_car=n_car, n_people=n_people)
    agent_info_history = []  # Inicializa una lista para almacenar la información de los agentes en cada paso
    for step in range(100):  # Ajusta el número de pasos según sea necesario
        mod.step()

        step_info = {"step": step}  # Crea un título para el paso actual
        agent_info_step = []  # Inicializa una lista para almacenar la información de los agentes en cada paso

        # Recopila la información importante de cada agente en cada paso
        for agent in mod.schedule.agents:
            agent_info = {}
            if isinstance(agent, carAgent) or isinstance(agent, peopleAgent):
                # Información para carAgent y peopleAgent
                agent_info["type"] = "carAgent" if isinstance(agent, carAgent) else "peopleAgent"
                agent_info["unique_id"] = agent.unique_id
                agent_info["position"] = agent.beliefs['position']
            elif isinstance(agent, StoplightAgent):
                # Información para StoplightAgent
                agent_info["type"] = "StoplightAgent"
                agent_info["unique_id"] = agent.unique_id
                agent_info["state"] = agent.state

            agent_info_step.append(agent_info)

        step_info["agents"] = agent_info_step  # Agrega la información del paso actual al título del paso
        agent_info_history.append(step_info)  # Agrega el título y la información al historial
    
    # Guarda la información de todos los pasos en un solo archivo JSON
    with open('agent_info_history.json', 'w') as json_file:
        json.dump(agent_info_history, json_file, indent=4)