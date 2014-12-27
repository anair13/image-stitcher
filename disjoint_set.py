class DisjointSet:
    """Implements a disjoint set with O(1) unions and (almost) O(1) finds"""
    def __init__(self, elements):
        self.tree = [-1] * len(elements) # array representation of a tree
        self.elements_to_index = {}
        self.elements = elements
        i = 0
        for e in elements:
            self.elements_to_index[e] = i
            i += 1

    def _root_union(self, root1, root2):
        """Combines sets of root1 and root2 in tree"""
        if root1 == root2:
            return
        if (self.tree[root1] < self.tree[root2]): # root2 is bigger
            self.tree[root2] += self.tree[root1]
            self.tree[root1] = root2
        else:
            self.tree[root1] += self.tree[root2]
            self.tree[root2] = root1

    def _find_index(self, k):
        """Returns root of tree containing k"""
        if self.tree[k] < 0:
            return k
        else:
            self.tree[k] = self._find_index(self.tree[k])
            return self.tree[k]

    def find(self, key):
        return self.elements[self._find_index(self.elements_to_index[key])]

    def union(self, key1, key2):
        self._root_union(self._find_index(self.elements_to_index[key1]),
                         self._find_index(self.elements_to_index[key2]))

    def get_sets(self):
        sets = {}
        for e in self.elements:
            sets.setdefault(self.find(e), set()).add(e)
        return sets.values()






