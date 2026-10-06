"""
CustomHeap: A heap implementation with quadruple keys (k1, k2, k3, k4)
Comparison priority: k1 > k2 > k3 > k4 (lexicographic ordering)
Supports both min-heap and max-heap modes
"""

import heapq


class HeapItem:
    """
    Wrapper class for heap items with quadruple key comparison
    """
    def __init__(self, k1, k2, k3, k4, value=None, reverse=False):
        """
        Args:
            k1, k2, k3, k4: Float values for comparison (in priority order)
            value: Optional data associated with this item
            reverse: If True, reverse comparison for max-heap behavior
        """
        self.k1 = k1
        self.k2 = k2
        self.k3 = k3
        self.k4 = k4
        self.value = value
        self.reverse = reverse
    
    def __lt__(self, other):
        """
        Less than comparison
        For min-heap: returns True if self < other
        For max-heap: returns True if self > other (reversed)
        Compares lexicographically: k1, then k2, then k3, then k4
        """
        if self.reverse:
            # Reversed comparison for max-heap
            if self.k1 != other.k1:
                return self.k1 > other.k1
            if self.k2 != other.k2:
                return self.k2 > other.k2
            if self.k3 != other.k3:
                return self.k3 > other.k3
            return self.k4 > other.k4
        else:
            # Normal comparison for min-heap
            if self.k1 != other.k1:
                return self.k1 < other.k1
            if self.k2 != other.k2:
                return self.k2 < other.k2
            if self.k3 != other.k3:
                return self.k3 < other.k3
            return self.k4 < other.k4
    
    def __le__(self, other):
        """Less than or equal"""
        return self < other or self == other
    
    def __gt__(self, other):
        """Greater than"""
        return not self <= other
    
    def __ge__(self, other):
        """Greater than or equal"""
        return not self < other
    
    def __eq__(self, other):
        """Equality (based on keys only, not reverse flag)"""
        return (self.k1 == other.k1 and 
                self.k2 == other.k2 and 
                self.k3 == other.k3 and 
                self.k4 == other.k4)
    
    def __ne__(self, other):
        """Not equal"""
        return not self == other
    
    def __repr__(self):
        """String representation"""
        return f"HeapItem(k1={self.k1}, k2={self.k2}, k3={self.k3}, k4={self.k4}, value={self.value})"
    
    def get_keys(self):
        """Return keys as tuple"""
        return (self.k1, self.k2, self.k3, self.k4)


class CustomHeap:
    """
    Custom heap with quadruple key (k1, k2, k3, k4) comparison
    Supports both min-heap and max-heap modes
    Items are ordered by k1, then k2, then k3, then k4 (lexicographically)
    """
    
    def __init__(self, heap_type='min'):
        """
        Initialize heap
        
        Args:
            heap_type: 'min' for min-heap (default) or 'max' for max-heap
        
        Raises:
            ValueError: If heap_type is not 'min' or 'max'
        """
        if heap_type not in ('min', 'max'):
            raise ValueError("heap_type must be 'min' or 'max'")
        
        self._heap = []
        self.heap_type = heap_type
        self.reverse = (heap_type == 'max')
    
    def push(self, k1, k2, k3, k4, value=None):
        """
        Push an item onto the heap
        
        Args:
            k1, k2, k3, k4: Float keys for comparison
            value: Optional associated data
        """
        item = HeapItem(k1, k2, k3, k4, value, reverse=self.reverse)
        heapq.heappush(self._heap, item)
    
    def pop(self):
        """
        Pop and return the top priority item from the heap
        For min-heap: returns smallest item
        For max-heap: returns largest item
        
        Returns:
            HeapItem: The top priority item
        
        Raises:
            IndexError: If heap is empty
        """
        return heapq.heappop(self._heap)
    
    def peek(self):
        """
        Return the top priority item without removing it
        For min-heap: returns smallest item
        For max-heap: returns largest item
        
        Returns:
            HeapItem: The top priority item
        
        Raises:
            IndexError: If heap is empty
        """
        return self._heap[0]
    
    def pushpop(self, k1, k2, k3, k4, value=None):
        """
        Push item and then pop the top priority item
        More efficient than push() followed by pop()
        
        Returns:
            HeapItem: The popped item
        """
        item = HeapItem(k1, k2, k3, k4, value, reverse=self.reverse)
        return heapq.heappushpop(self._heap, item)
    
    def replace(self, k1, k2, k3, k4, value=None):
        """
        Pop top priority item and then push new item
        More efficient than pop() followed by push()
        
        Returns:
            HeapItem: The popped item
        """
        item = HeapItem(k1, k2, k3, k4, value, reverse=self.reverse)
        return heapq.heapreplace(self._heap, item)
    
    def __len__(self):
        """Return number of items in heap"""
        return len(self._heap)
    
    def __bool__(self):
        """True if heap is non-empty"""
        return len(self._heap) > 0
    
    def is_empty(self):
        """Check if heap is empty"""
        return len(self._heap) == 0
    
    def clear(self):
        """Remove all items from heap"""
        self._heap.clear()
    
    def get_all(self):
        """
        Return all items as a list (not in sorted order)
        """
        return list(self._heap)
    
    def get_sorted(self):
        """
        Return all items in priority order
        For min-heap: smallest to largest
        For max-heap: largest to smallest
        WARNING: This empties the heap!
        """
        sorted_items = []
        while self._heap:
            sorted_items.append(heapq.heappop(self._heap))
        return sorted_items
    
    def get_sorted_copy(self):
        """
        Return all items in priority order without modifying the heap
        For min-heap: smallest to largest
        For max-heap: largest to smallest
        """
        return sorted(self._heap)
    
    def __repr__(self):
        """String representation"""
        return f"CustomHeap(type={self.heap_type}, size={len(self._heap)})"

    def remove_by_value(self, value):
        """
        Remove and return an item from the heap by its value
        
        Args:
            value: The value to search for and remove
        
        Returns:
            HeapItem or None: The removed item if found, None otherwise
        
        Time Complexity: O(n) for search + O(n) for heapify = O(n)
        """
        # Find and remove the item
        for i, item in enumerate(self._heap):
            if item.value == value:
                # Found the item
                removed_item = self._heap[i]
                
                # Replace with last item
                last_item = self._heap.pop()
                
                # If we didn't just remove the last item, restore heap property
                if i < len(self._heap):
                    self._heap[i] = last_item
                    heapq.heapify(self._heap)
                
                return removed_item
        
        # Not found
        return None

# Example usage and testing
if __name__ == "__main__":
    print("="*60)
    print("MIN-HEAP EXAMPLE")
    print("="*60)
    
    # Create min-heap
    min_heap = CustomHeap(heap_type='min')
    
    # Add items
    min_heap.push(1.5, 2.0, 3.0, 4.0, value="A")
    min_heap.push(1.5, 2.0, 2.5, 4.0, value="B")  # Smaller k3
    min_heap.push(1.5, 1.5, 3.0, 4.0, value="C")  # Smaller k2
    min_heap.push(2.0, 1.0, 1.0, 1.0, value="D")  # Larger k1
    min_heap.push(1.5, 2.0, 3.0, 3.5, value="E")  # Smaller k4
    
    print(f"Heap: {min_heap}")
    print(f"Heap empty? {min_heap.is_empty()}")
    
    # Peek at smallest
    smallest = min_heap.peek()
    print(f"\nTop priority (peek): {smallest}")
    
    # Pop all items in order
    print("\nPopping items in order (smallest to largest):")
    while min_heap:
        item = min_heap.pop()
        print(f"  {item.get_keys()} -> value: {item.value}")
    
    print("\n" + "="*60)
    print("MAX-HEAP EXAMPLE")
    print("="*60)
    
    # Create max-heap
    max_heap = CustomHeap(heap_type='max')
    
    # Add same items
    max_heap.push(1.5, 2.0, 3.0, 4.0, value="A")
    max_heap.push(1.5, 2.0, 2.5, 4.0, value="B")
    max_heap.push(1.5, 1.5, 3.0, 4.0, value="C")
    max_heap.push(2.0, 1.0, 1.0, 1.0, value="D")  # Largest k1
    max_heap.push(1.5, 2.0, 3.0, 3.5, value="E")
    
    print(f"Heap: {max_heap}")
    
    # Peek at largest
    largest = max_heap.peek()
    print(f"\nTop priority (peek): {largest}")
    
    # Pop all items in order
    print("\nPopping items in order (largest to smallest):")
    while max_heap:
        item = max_heap.pop()
        print(f"  {item.get_keys()} -> value: {item.value}")
    
    print("\n" + "="*60)
    print("COMPARISON TEST")
    print("="*60)
    
    # Test with priority on different keys
    test_heap = CustomHeap(heap_type='max')
    test_heap.push(5.0, 3.0, 2.0, 1.0, value="First")
    test_heap.push(5.0, 3.0, 2.0, 2.0, value="Second")  # k4 differs
    test_heap.push(5.0, 3.0, 1.0, 10.0, value="Third")  # k3 smaller
    test_heap.push(5.0, 4.0, 0.0, 0.0, value="Fourth")  # k2 larger
    
    print("Max-heap order (largest k values first):")
    for item in test_heap.get_sorted_copy():
        print(f"  {item}")