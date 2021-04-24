#include <iostream>
#include <fstream>
#include <algorithm>

int main()
{
    int var_this;
    std::ifstream cFile ("config/test.ini");
    if (cFile.is_open())
    {
        std::string line;
        while(getline(cFile, line)){
            line.erase(std::remove_if(line.begin(), line.end(), isspace),
                                 line.end());
            if(line[0] == '#' || line.empty())
                continue;
            auto delimiterPos = line.find("=");
            auto name = line.substr(0, delimiterPos);
            auto value = line.substr(delimiterPos + 1);

	    if (name == "this") {
		var_this = stoi(value);
	        std::cout << "we have this " << var_this << std::endl;

	    }

            std::cout << name << " " << value << '\n';
        }
        
    }
    else {
        std::cerr << "Couldn't open config file for reading.\n";
    }
}
