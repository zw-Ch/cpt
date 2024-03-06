#pragma once

#include <vector>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <iostream>
#include "utils.h"

std::vector<std::vector<float>> read_csv(const std::string &file_path, const std::vector<std::string> &columns = {},
                                         bool show = false);
