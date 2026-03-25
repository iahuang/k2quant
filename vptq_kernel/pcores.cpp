#include <thread>

#ifdef __linux__
#include <fstream>
#include <set>
#include <string>

int physical_cores() {
    std::ifstream f("/proc/cpuinfo");
    if (!f) return std::thread::hardware_concurrency();

    std::set<std::pair<int,int>> cores;
    std::string line;
    int physical_id = 0, core_id = 0;

    while (std::getline(f, line)) {
        if (line.rfind("physical id", 0) == 0)
            physical_id = std::stoi(line.substr(line.find(':') + 1));
        else if (line.rfind("core id", 0) == 0)
            core_id = std::stoi(line.substr(line.find(':') + 1));
        else if (line.empty())
            cores.insert({physical_id, core_id});
    }

    return cores.empty() ? std::thread::hardware_concurrency() : static_cast<int>(cores.size());
}

#else

int physical_cores() {
    return std::thread::hardware_concurrency();
}

#endif