#include <iostream>
#include <fstream>
#include "nlohmann/json.hpp"
#include <map>
#include <set>
#include <cstdlib>

using json = nlohmann::json;
using namespace std;

int main() {
    system("python agents.py");
    ifstream json_file("agent_info_history.json");
    if (!json_file.is_open()) {
        cerr << "Failed to open JSON file." << endl;
        return 1;
    }

    json data;
    json_file >> data;

    map<int, string> first_instance; // Map to store the first instance of each agent

    for (const auto& step_data : data) {
        int step = step_data["step"];
        cout << "Step " << step << ":" << endl;

        set<int> current_step_agents; // Set to store agents in the current step

        for (const auto& agent_info : step_data["agents"]) {
            int unique_id = agent_info["unique_id"];
            current_step_agents.insert(unique_id);

            if (first_instance.find(unique_id) == first_instance.end()) {
                // If this is the first instance, store it in first_instance
                first_instance[unique_id] = "Agent " + to_string(unique_id);
                cout << first_instance[unique_id] << " was created" << endl;
            } else {
                cout << "Agent " << unique_id;
            }

            // Check if "position" exists and is not null
            if (agent_info.find("position") != agent_info.end() && !agent_info["position"].is_null()) {
                int x = agent_info["position"][0];
                int y = agent_info["position"][1];
                cout << " moved to (" << x << "," << y << ")";
            } else {
                cout << " has no position";
            }

            if (agent_info.find("state") != agent_info.end()) {
                string state = agent_info["state"];
                cout << " changed state to " << state;
            }

            cout << endl;
        }

        // Display last instances of agents that are not in the current step and mark them as destroyed
        for (const auto& entry : first_instance) {
            if (current_step_agents.find(entry.first) == current_step_agents.end()) {
                cout << entry.second <<" was destroyed" << endl;
            }
        }

        cout << endl;
    }

    return 0;
}
