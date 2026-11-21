// Build (MinGW):  g++ -std=c++17 -O2 example.cpp -I../include -L<target>/release -lweightdb -o example
// Usage:          example <repo> <rev> <tensor>
#include "weightdb.hpp"

#include <cmath>
#include <cstdio>
#include <iostream>

int main(int argc, char** argv) {
    if (argc < 4) {
        std::cerr << "usage: example <repo> <rev> <tensor>\n";
        return 2;
    }
    try {
        auto db = weightdb::Database::open(argv[1]);
        std::cout << "weightdb " << wdb_version() << ", " << argv[2] << " = " << db.resolve(argv[2]).substr(0, 12)
                  << "\n";

        auto info = db.tensor_info(argv[2], argv[3]);
        std::cout << argv[3] << " " << info.dtype << " [";
        for (size_t i = 0; i < info.shape.size(); ++i) std::cout << (i ? ", " : "") << info.shape[i];
        std::cout << "] " << info.nbytes << " bytes\n";

        if (info.dtype == "F32") {
            auto w = db.read_tensor<float>(argv[2], argv[3]);
            double sum = 0, sq = 0;
            for (float v : w) {
                sum += v;
                sq += double(v) * v;
            }
            std::printf("mean=%.6e l2=%.6e first=%.6f\n", sum / w.size(), std::sqrt(sq), w[0]);
            std::printf("read_weight[0]=%.6f\n", db.read_weight(argv[2], argv[3], 0));
        }

        std::cout << db.sql_pretty(
                         "SELECT role, count(*) AS tensors, sum(nbytes) AS bytes FROM tensors "
                         "WHERE commit_id = rev('" + std::string(argv[2]) + "') GROUP BY role ORDER BY bytes DESC")
                  << "\n";

        try {
            db.read_weight(argv[2], "no.such.tensor", 0);
        } catch (const weightdb::Error& e) {
            std::cout << "expected error: " << e.what() << "\n";
        }
    } catch (const weightdb::Error& e) {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }
    return 0;
}
