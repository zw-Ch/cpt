#include "process.h"
#include <vector>

std::string getFileName(const std::string &path)
{
    size_t found = path.rfind('/');
    if (found != std::string::npos)
    {
        return path.substr(found + 1);
    }
    else
    {
        return path;
    }
}

std::vector<size_t> getColumnIdx(const std::string &line, const std::vector<std::string> &columns)
{
    std::vector<size_t> idxs;
    std::istringstream iss(line);
    std::string column;
    size_t idx = 0;

    while (std::getline(iss, column, ','))
    {
        auto it = std::find(columns.begin(), columns.end(), column);
        if (it != columns.end())
        {
            idxs.push_back(idx);
        }
        idx++;
    }
    return idxs;
}

std::vector<float> getValueByIdx(const std::string &line, const std::vector<size_t> &idxs)
{
    std::vector<float> value;
    auto words = split(line, ',');
    for (const auto &idx : idxs)
    {
        value.push_back(std::stof(words[idx]));
    }
    return value;
}

std::vector<std::vector<float>> read_csv(const std::string &path, const std::vector<std::string> &columns,
                                         bool show)
{
    std::vector<size_t> idxs;
    std::vector<float> value;
    std::vector<std::vector<float>> values;

    std::string line;
    std::ifstream csv_file(path);
    if (csv_file.is_open())
    {
        std::cout << "========== "
                  << "Reading " << getFileName(path) << " ==========" << std::endl;
        int idx_row = 0;
        while (std::getline(csv_file, line))
        {
            if (idx_row == 0)
            {
                // std::cout << line << std::endl;
                idxs = getColumnIdx(line, columns);
            }
            else
            {
                try
                {
                    value = getValueByIdx(line, idxs);
                    values.push_back(value);
                }
                catch (const std::invalid_argument &e)
                {
                    // std::cout << "Error Line: " << line << std::endl;
                }
            }
            idx_row++;
        }
        // std::cout << "CSV Size = " << values.size() << std::endl;
        
        if (show) {
            print(columns);
            for (int i = 0; i < 5; i++) {
                print(values[i]);
            }
        }
    }
    else
    {
        throw std::runtime_error("Could not read file at path: " + path);
    }

    return values;
}
