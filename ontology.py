from owlready2 import *

#custom datatype for pairs of integers (coordinates)
class Coordinate(object):
    def __init__(self, value):
        self.value = value

def pair_parser(s):
    values = s.split(',')
    return Coordinate((int(values[0]), int(values[1])))

def pair_unparser(x):
    return "{},{}".format(x.value[0], x.value[1])

# Register the custom datatype
declare_datatype(Coordinate, "http://www.example.org/custom#Coordinate", pair_parser, pair_unparser)

#Creación de nueva ontologia
base_iri = "http://ontology.example.com/urban-mobility"
onto = get_ontology(base_iri)

# classes and subclasses
with onto:
    class Agent(Thing):
        pass

    class MobileAgent (Agent):
        pass

    class ImmobileAgent (Agent):
        pass

# subclasses agents
with onto:
    class PeopleAgent(MobileAgent):
        pass
    
    class CarAgent(MobileAgent):
        pass

    class StopLightAgent(ImmobileAgent):
        pass
    
    class AuctionAgent(ImmobileAgent):
        pass

#properties of agents
with onto: 
    class model(CarAgent >> str): #unique identifier for car agent
        pass 

    class destination(CarAgent >> Coordinate): 
        pass  

    class destination(PeopleAgent >> Coordinate):
       pass
    
    class frontDistance(CarAgent >> int): #distance of closest object infront of the agent
        pass

    class frontDistance(PeopleAgent >> int):  #distance of closest object infront of the agent
        pass

    class position(CarAgent >> Coordinate):
        pass

    class position(PeopleAgent >> Coordinate):
        pass

    class id(PeopleAgent >> str): #unique identifier for people agent
        pass

    class state(StopLightAgent >> str):  #StopLight can be red, yellow or green
        pass

    class waitTime(StopLightAgent >> int):  #tiempo de espera en rojo
        pass 

    class carAmount(StopLightAgent >> int):   #cantidad de carros en su línea
        pass

    class winningStopLight(AuctionAgent >> str): #semaforo ganador de la subasta
        pass

    class auctionActive(AuctionAgent >> bool): #es posible realizar la subasta?
        pass

    class time(AuctionAgent >> int): #tiempo que se asigna al semaforo ganador
        pass

#instances of agents
trafficLight = StopLightAgent()
person =  PeopleAgent()
truck = CarAgent()
auctioneer = AuctionAgent()


#Add properties 
person.id.append(1)
person.destination.append(Coordinate((20, 20)))
person.frontDistance.append(0)
person.position.append(Coordinate((1, 2)))
truck.position.append(Coordinate((3, 3)))
truck.frontDistance.append(0)
truck.model.append("vocho")
truck.destination.append(Coordinate((20, 20)))
trafficLight.state.append("green")
trafficLight.waitTime.append(10)
trafficLight.carAmount.append(9)
auctioneer.winningStopLight.append(1)
auctioneer.auctionActive.append(True)
auctioneer.time.append(10)


#Save ontology
onto.save(file="ontology.owl", format = "rdfxml")
