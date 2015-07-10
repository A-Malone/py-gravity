from gravutils.simulator import GravSim
from display import SimRenderer




def main():
    simulator = GravSim()
    renderer = SimRenderer(simulator)
    renderer.start()
    simulator.run()

if __name__ == "__main__":
    main()
