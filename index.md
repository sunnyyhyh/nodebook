# Sunnyの刷题笔记

## **用到的python语法**

## **通用模型**

### 二分法

```python
def erfen(arr, left, right, traget):
    while left < right:
        # 这样写能够防止大数溢出
        mid = left + (left-right)//2
        # 左+1 右不变，左闭右开
        if arr[mid] < target:
            left = mid + 1
        else:
            right = mid
    # 因为采用left<right，所以left==right，返回谁都可以
    return left
```

```python
def erfen(arr, left, right, traget):
    # 这样做的好处是能找到超过数组范围的位置
    while left <= right:
        mid = left + (left-right)//2
        if arr[mid] < target:
            left = mid + 1
        # 不考虑target有重复值
        else:
            right = mid - 1
    # 但在返回时left>right，需要考虑返回谁
    return left
```



## **剑指offer**

**参考[力扣](https://leetcode-cn.com/)中的[图解算法数据结构](https://leetcode-cn.com/leetbook/detail/illustration-of-algorithm/)与剑指offer实体书(第二版)**

感谢Krahets的分享总结~！

### **数据结构**

***

### **动态规划**

***

### **搜索与回溯**

***

### **分治**

***

### **排序**

***

上面的二刷再补全

#### [**剑指 Offer 45. 把数组排成最小的数**](https://leetcode-cn.com/problems/ba-shu-zu-pai-cheng-zui-xiao-de-shu-lcof/)  

> 自己思路：  

- 无 

> 解题思路：  
>
> - 若拼接字符串 x+y > y+x , 则x大于y  
> - 反之，若 x+y < y+x , 则x小于y   

新建函数取代“>” “<”  
其他照搬sort实现

#### [**剑指 Offer 61. 扑克牌中的顺子**](https://leetcode-cn.com/problems/bu-ke-pai-zhong-de-shun-zi-lcof/)  

> 自己思路：  
>
> - 满足如下条件则为顺子：  
>   max-min<5 and len(不重复值)==5
> - 再考虑到0的特殊情况，通过0减少不重值个数即可  

新建函数取代“>” “<”  
其他照搬sort实现

> 另解：

- 先sort对数组进行排序
- 然后遍历按前后相等关系判断重复 


### **查找**

***

#### [剑指 Offer 03. 数组中重复的数字](https://leetcode-cn.com/problems/shu-zu-zhong-zhong-fu-de-shu-zi-lcof/)

> 自己思路:

使用sort排序，遍历查找

时间复杂度O(NlogN) 空间复杂度O(1)

```python
def findRepeatNumber(self, nums: List[int]) -> int:
    nums.sort()
    for i, _ in enumerate(nums[:-1]):
        if nums[i] == nums[i+1]:
            return nums[i]
```

当然也可以用set()来记录

时间复杂度O(N) 空间复杂度O(N)

```python
def findRepeatNumber(self, nums: List[int]) -> int:
    a = set()
    for n in nums:
        if n in a:
            return n
        else:
            a.add(n)
```

> 优化思路：

原地交换

时间复杂度O(N) 空间复杂度O(1)

因为题目中有数组长为n，数字在0~n-1的约束条件

所以，若无重复则可以将列表下标与值一一对应，如nums[0]=0,nums[1]=1

因其特殊性，可以使用遍历-原地交换的方法对其进行排序，排序过程中寻找重复值剪枝

```python
def findRepeatNumber(self, nums: List[int]) -> int:
    i = 0
    while i < len(nums):
        if nums[i] == i:
            i += 1
            continue
            if nums[nums[i]] == nums[i]:
                return nums[i]
            nums[nums[i]], nums[i] = nums[i], nums[nums[i]]
```

> 附：注意赋值语句前后有相同元素时的修改顺序(最后一行的顺序反过来会报错)



> 附题：**不修改数组找出重复数字**

注意题干，若果是需修改数组找出重复数字，数组长度应比数字范围多1

>  思路：

对允许的数字范围进行二分，并计算该范围内数字在原数组出现次数

若出现次数多于范围长度，则说明有重复数字在此范围中，继续二分查找

```python
def getcount(left, right):
            count = 0
            for n in nums:
                if left <= n <= right:
                    count += 1
            return count

        start, end = 0, len(nums)-1
        while start<=end:
            mid = (start+end) // 2 
            count =  getcount(start, mid)
            if start == end:
                if count > 1:
                    return start
                else:
                    return 
            if count > mid-start+1:
                end = mid
            else:
                start = mid+1
        return 
```



#### [剑指 Offer 04. 二维数组中的查找](https://leetcode-cn.com/problems/er-wei-shu-zu-zhong-de-cha-zhao-lcof/)

> 自己思路：暴力法0(n*m)

> 优化思路：

从右上或左下出发，以右上为例

- 若等于target则输出
- 若大于target则左移
- 若小于target则下移

可以将这个过程视作将二维数组转化为以右上为根的排序二叉树(左下右大)

```python
def findNumberIn2DArray(self, matrix: List[List[int]], target: int) -> bool:
    if not matrix:
        return False
    w, h = len(matrix[0]), len(matrix)
    i, j = 0, w-1
    while i<h and j>=0:
        if matrix[i][j] == target:
            return True
        if matrix[i][j] > target:
            j -= 1
        else:
            i += 1
    return False
```



#### [剑指 Offer 11. 旋转数组的最小数字](https://leetcode-cn.com/problems/xuan-zhuan-shu-zu-de-zui-xiao-shu-zi-lcof/)

> 自己思路：

最简单的肯定是sort后取nums[0]，时间复杂度O(NlogN)

旋转数组可以被分为递增的两个部分，而分界值为数组中的最后一个数

找最小数明显是个查找类问题，使用二分查找肯定没错

于是就有在二分查找的基础上，以nums[-1]作为判断条件搜索mid属于哪部分

但还存在一个问题，当 mid == nums[-1] 时如何处理？

> 解决：

当 mid == nums[-1]时可以视为最小值重复，所以有边界-1也不影响(相当于去掉了Nums[-1])

当然也可以直接输出当前区间了

空间复杂度O(longN)

```python
def minArray(self, numbers: List[int]) -> int:
    left, right = 0, len(numbers)-1
    while left<right:
        mark = numbers[right]
        mid = (left+right)//2
        if numbers[mid] > mark:
            left = mid+1
        elif numbers[mid] < mark:
            right = mid
        else:
            # right -= 1
            return min(numbers[left:right])
    return numbers[left]
```



#### [剑指 Offer 50. 第一个只出现一次的字符](https://leetcode-cn.com/problems/di-yi-ge-zhi-chu-xian-yi-ci-de-zi-fu-lcof/)

> 自己思路：

没啥可说的，字典哈希表就完了

> 优化思路：

代码能更好看

```python
dic = {}
for c in s:
	dic[c] = not c in dic
for c in s:
    if dic[c]: 
    	return c
    return " "
```



#### [剑指 Offer 53 - I. 在排序数组中查找数字 I](https://leetcode-cn.com/problems/zai-pai-xu-shu-zu-zhong-cha-zhao-shu-zi-lcof/)

> 自己思路：

分两次分别查找target的左右两个端点(外边那个点)

将二分查找法写两次就完事

```python
def search(self, nums: List[int], target: int) -> int:
	left, right = 0, len(nums)-1
	# 这里使用<=循环找左端点，后边取值应取right(因为找的点比target小，所哟循环结束后right<left)
    while left <= right: 
        mid = (left+right)//2
        if nums[mid] >= target:
        	right = mid-1
        else:
        	left = mid+1
    start = right
    left, right = 0, len(nums)-1
    while left <= right:
    	mid = (left+right)//2
    	if nums[mid] <= target:
        	left = mid+1
        else:
        	right = mid-1
    # 这里同理
    end = left
    return end-start-1
```



> 优化思路：

写两次二分查找肯定存在冗余，提取出来写成函数

函数为提取右端点，左端点使用target-1查找，这样势必会找到target最左端，所以最后不用-1

```python
def search(self, nums: [int], target: int) -> int:
    def helper(tar):
        i, j = 0, len(nums) - 1
        while i <= j:
            m = (i + j) // 2
            if nums[m] <= tar: i = m + 1
            else: 
                j = m - 1
        return i
    return helper(target) - helper(target - 1)
```



#### [剑指 Offer 53 - II. 0～n-1中缺失的数字](https://leetcode-cn.com/problems/que-shi-de-shu-zi-lcof/)

> 自己思路：

二分查找，更新left的方式换为序号与值相匹配

```python
def missingNumber(self, nums: List[int]) -> int:
    i, j = 0, len(nums) - 1
    while i <= j:
        m = (i + j) // 2
        if nums[m] == m: 
            i = m + 1
        else: 
            j = m - 1
    return i
```





### **双指针**

***

#### [剑指 Offer 18. 删除链表的节点](https://leetcode-cn.com/problems/shan-chu-lian-biao-de-jie-dian-lcof/)

> 自己思路：

链表删除节点需要记录前一个和当前两个节点

使用双指针，一个记录pre，一个记录cur

```python
def deleteNode(self, head: ListNode, val: int) -> ListNode:
	if head.val == val:
        return head.next
    pre, cur = head, head.next
    while cur:
        if cur and cur.val == val:
            pre.next = cur.next
            break
        pre = cur
        cur = cur.next
    return head
```

> 优化思路：

将判断条件放进循环，跳出循环直接执行删除节点操作

```python
def deleteNode(self, head: ListNode, val: int) -> ListNode:
	if head.val == val:
        return head.next
    pre, cur = head, head.next
    while cur and cur.val != val:
        pre = cur
        cur = cur.next
    if cur:
        pre.next = cur.next
    return head
```



#### [剑指 Offer 21. 调整数组顺序使奇数位于偶数前面](https://leetcode-cn.com/problems/diao-zheng-shu-zu-shun-xu-shi-qi-shu-wei-yu-ou-shu-qian-mian-lcof/)

> 自己思路：

双指针，从前往后搜索偶数，从后往前搜索奇数，每次搜索到了互换

```python
def exchange(self, nums: List[int]) -> List[int]:
    left, right = 0, len(nums)-1
    while left < right:
        while left<right and nums[left]%2 == 1:
            left += 1
        while left<right and nums[right]%2 == 0:
            right -= 1
        nums[left], nums[right] = nums[right], nums[left]
    return nums
```

> 优化思路：

可以用&1替换除余判断奇偶数

原理是：&1会自动补全为&00001，相当于只计算最后一位&1，而二进制区分奇偶数正在于最后一位是0还是1

```python
def exchange(self, nums: List[int]) -> List[int]:
    left, right = 0, len(nums)-1
    while left < right:
        while left<right and nums[left]&1 == 1:
            left += 1
        while left<right and nums[right]&1 == 0:
            right -= 1
        nums[left], nums[right] = nums[right], nums[left]
    return nums
```

当然，考虑到对功能的抽象，可以把判断条件单独写为函数

```python
def isEven(num):
	return num & 1 == 0
    left, right = 0, len(nums)-1
    while left < right:
    	while left<right and not isEven(nums[left]):
            left += 1
        while left<right and isEven(nums[right]):
            right -= 1
        nums[left], nums[right] = nums[right], nums[left]
    return nums
```



#### [剑指 Offer 22. 链表中倒数第k个节点](https://leetcode-cn.com/problems/lian-biao-zhong-dao-shu-di-kge-jie-dian-lcof/)

> 自己思路：

先遍历一遍找到链表的长度，再遍历找到输出位置

```python
def getKthFromEnd(self, head: ListNode, k: int) -> ListNode:
    node, len = head, 1
    while node.next:
        len += 1
        node = node.next
    for _ in range(len-k):
        head = head.next
    return head
```



>优化思路：

可以使用双指针，避免统计链表长度

```python
def getKthFromEnd(self, head: ListNode, k: int) -> ListNode:    front, behind = head, head    for _ in range(k):        behind = behind.next    while behind:        front, behind = front.next, behind.next    return front
```



> 补充：

虽然以上能够A了题，但那是因为测试样本不够鲁棒，真实情况还有很多条件需要考虑

```python
def getKthFromEnd(self, head: ListNode, k: int) -> ListNode:    if not head or not k:        return    front, behind = head, head    for _ in range(k):        if not behind.next:            return         behind = behind.next    while behind:        front, behind = front.next, behind.next    return front
```



#### [剑指 Offer 25. 合并两个排序的链表](https://leetcode-cn.com/problems/he-bing-liang-ge-pai-xu-de-lian-biao-lcof/)

> 自己思路：

双指针分别遍历两个链表，判断大小后依次插入输出链表中，唯一需要注意的是输出是out.next

遍历链表时间复杂度O(M+N)，其实只引入了一个指向头的节点，空间复杂度O(1)

```python
def mergeTwoLists(self, l1: ListNode, l2: ListNode) -> ListNode:    out = tem = ListNode(0)    while l1 and l2:        if l1.val < l2.val:            tem.next, l1 = l1, l1.next        else:            tem.next, l2 = l2, l2.next        tem = tem.next    if l1:        tem.next = l1    if l2:        tem.next = l2    # tem.next = l1 if l1 else l2    return out.next
```



#### [剑指 Offer 52. 两个链表的第一个公共节点](https://leetcode-cn.com/problems/liang-ge-lian-biao-de-di-yi-ge-gong-gong-jie-dian-lcof/)

> 需要注意的事：

公共节点意味着两个链接有公共部分，所以判断条件为 A == B

> 自己思路：

一次完全遍历找长度，然后按差值设置双指针分别遍历A,B，直到A==B，时间复杂度O(M+N)

```python
def getIntersectionNode(self, headA: ListNode, headB: ListNode) -> ListNode:    markhead = ListNode(0)    A, B, lA, lB, mark = headA, headB, 0, 0, None    while A = A.next:        lA += 1        A = A.next    while B:        lB += 1        B = B.next    if lA > lB:        for _ in range(lA-lB):            headA = headA.next    else:        for _ in range(lB-lA):            headB = headB.next    while headA:        if headA == headB:            return headA        headA = headA.next         headB = headB.next    return 
```

> 优化思路：

分别对A在前B在后与B在前A在后的链表进行遍历，直到A==B即可，时间复杂度O(M+N)

```python
A, B = headA, headB    while A != B:    	A = A.next if A else headB    	B = B.next if B else headA    return A
```



#### [剑指 Offer 57. 和为s的两个数字](https://leetcode-cn.com/problems/he-wei-sde-liang-ge-shu-zi-lcof/)

> 自己思路：

最朴素的是循环查找，但是超时了

得考虑利用排序的条件，设置双指针分别从前后开始遍历数组直到满足条件，时间复杂度O(N)

```python
def twoSum(self, nums: List[int], target: int) -> List[int]:	left, right = 0, len(nums)-1    while left < right:        while nums[left]+nums[right]>target:            right -= 1        while nums[left]+nums[right]<target:            left += 1        if nums[left]+nums[right] == target:            return [nums[left], nums[right]]        return 
```



#### [剑指 Offer 58 - I. 翻转单词顺序](https://leetcode-cn.com/problems/fan-zhuan-dan-ci-shun-xu-lcof/)

> 自己思路：

本来考虑用空间复杂度O(1)的方法，但太麻烦了debug吐了

改用从后向前遍历的方法，这样要新建一个字符串按顺序存放

思路很简单，但真的很不擅长遍历数组啊喂，各种越界:cry:

教训是先把各种情况想好，每个mark指向哪里什么位置想好再写，不然改起来会很麻烦

时间复杂度O(N)，空间复杂度O(N)

```python
def reverseWords(self, s: str) -> str:	s = s.strip()    s = list(s)    out = ""    front, behind = len(s)-1, len(s)-1    while front>=0:        while front>=0 and s[front] != " ":        	front -= 1        out += ' '+ ''.join(s[front+1:behind+1])        while s[front] == " ":            front -= 1            behind = front    return out.strip()
```



> 优化思路：

用python内置函数...虽然笔试题这么写不好，但还是希望能想起来，平时用真的很方便

```python
def reverseWords(self, s: str) -> str:	s = s.strip()	s = s.split()	s.reverse() 	return ' '.join(s)
```



#### [剑指 Offer 15. 二进制中1的个数](https://leetcode-cn.com/problems/er-jin-zhi-zhong-1de-ge-shu-lcof/)

> 自己思路：

托之前奇偶数题的福，还是能想到使用&1判断最后一位是否是1

然后再使用>>逐位检查

```python
def hammingWeight(self, n: int) -> int:    num = 0    while n:        if n & 1:            num += 1        n >>= 1    return num
```

> 优化思路：

竟然还是有特殊技巧，使用 n & n-1可以直接干掉最后一个1，重复几次就是干掉了一个1

```python
def hammingWeight(self, n: int) -> int:    while n:        num += 1        n &= n-1    return num
```

 

#### [剑指 Offer 56 - I. 数组中数字出现的次数](https://leetcode-cn.com/problems/shu-zu-zhong-shu-zi-chu-xian-de-ci-shu-lcof/)

> 自己没有思路

> 题解：

用了不少位运算的原理，记录如下：

+ 0^x = x，所以0可以用来初始化异或的值而不用担心影响后续运算
+ x^x = 0，所以异或可以用于干掉成双的值，保留单身的值 ~~老FFF团员了~~
+ 故技重施的使用&找1的位置

```python
def singleNumbers(self, nums: List[int]) -> List[int]:    tem, mark = 0, 1    for n in nums:        tem ^= n    while not tem&mark:        mark <<= 1    l1, l2 = 0, 0    for n in nums:        if n&mark:            l1 ^= n        else:            l2 ^= n    return l1, l2 
```



#### [剑指 Offer 56 - II. 数组中数字出现的次数 II](https://leetcode-cn.com/problems/shu-zu-zhong-shu-zi-chu-xian-de-ci-shu-ii-lcof/)

> 自己思路没有

> 题解

统计数组中每个数的二进制每位之和，和能整除3则该位置为输出数的0，否则为1

```python
def singleNumber(self, nums: List[int]) -> int:    counts = [0] * 32    for num in nums:        for i in range(32):            counts[i] += num & 1            num >>= 1    tem, out = 1, 0    for i, c in enumerate(counts):        out += tem*(c%3)        tem <<= 1    return out
```

另有使用位运算的解法...人看傻了不学了之后再说吧



### **位运算**

***

### **数学**

***

### **模拟**

***

## **other**

