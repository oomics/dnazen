#include <string>
#include <vector>

template <typename T>
std::vector<int> cal_levenshtein_distance(const std::vector<T> &s1,
                                          const std::vector<T> &target) {
    size_t m = s1.size();
    size_t n = target.size();

    std::vector<std::vector<int>> dp(m + 1, std::vector<int>(n + 1));

    for (size_t i = 0; i < m + 1; i++) {
        dp[i][0] = i;
    }

    for (size_t j = 0; j < n + 1; j++) {
        dp[0][j] = j;
    }

    // Fill dp state
    for (size_t i = 1; i <= m; ++i) {
        for (size_t j = 1; j <= n; ++j) {
            int cost = (s1[i - 1] == target[j - 1]) ? 0 : 1;
            dp[i][j] = std::min({
                dp[i - 1][j] + 1,        // Deletion
                dp[i][j - 1] + 1,        // Insertion
                dp[i - 1][j - 1] + cost  // Substitution
            });
        }
    }

    return dp[m];
}