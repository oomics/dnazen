#include"common.hpp"

int main() {
    ConcurrentHashMap<int, size_t> hashmap(8);

    hashmap[0] += 2;
    hashmap[1] = 3;
    hashmap[0] += 2;
    std::cout<<hashmap[0]<<std::endl;
    std::cout<<hashmap[1]<<std::endl;

    std::unordered_map<int, size_t> map;
    std::cout<<"dumping..."<<std::endl;
    hashmap.to_unordered_map(map);
    std::cout<<"unordered_map:"<<std::endl;
    std::cout<<map[0]<<std::endl;
    std::cout<<map[1]<<std::endl;
}
