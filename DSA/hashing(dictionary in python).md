### The following is to understand the use of hashing(aka dict in python)

#### leetcode problem : https://leetcode.com/problems/group-anagrams/

```python

class Solution:
    def groupAnagrams(self, strs: List[str]) -> List[List[str]]:
        anagram_dict = dict()

        for i in strs:
            sorted_i = "".join(sorted(i))
            if sorted_i not in anagram_dict:
                anagram_dict[sorted_i] = []
            anagram_dict[sorted_i].append(i)
            
        return(list(anagram_dict.values()))

```
