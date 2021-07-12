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

感谢Krahets的分享总结，K神，yyds！ :happy:

以下代码都是在力扣网上A过的，但不排除个别复制错了或者缩进错了:cry:

***

### **数据结构**

***

#### [剑指 Offer 05. 替换空格](https://leetcode-cn.com/problems/ti-huan-kong-ge-lcof/)

> 自己思路

分两步，首先计算替换后的长度，用于构建新数组或扩展数组。然后遍历替换

最简单的方法是直接构建新数组然后顺序写入，但需要O(N)的空间复杂度

如果在原地操作，每次遇到空格都要讲之后的字符向后移动，时间复杂度O(N^2)

为了达到时间复杂度O(N)，空间复杂度降到O(1)，可以将原字符串扩展然后在原地修改

道理虽然是这么讲的，但python并不能以O(1)替换字符串里的字符啊，所以最后还是转了list，而这个list为O(N)空间复杂度

当然，在原始字符串上移动需要记录原始与移动后的位置，也是个双指针问题

```python
class Solution:
    def replaceSpace(self, s: str) -> str:
        space_num = s.count(' ')
        out = list(s + space_num * 2 * ' ')
        i, j = len(s)-1, len(out)-1
        while i > -1:
            if out[i] == ' ':
                out[j-2:j+1] = '%20'
                j -= 3
            else:
                out[j] = out[i]
                j -= 1
            i -= 1
        return "".join(out)
```



#### [剑指 Offer 06. 从尾到头打印链表](https://leetcode-cn.com/problems/cong-wei-dao-tou-da-yin-lian-biao-lcof/)

> 自己思路

对于链表有个常识是：完整走一遍是少不了的

既然返回的只是值那从头到尾记录val，输出时翻转就好了

时间复杂度O(N)，空间复杂度O(N)

```python
class Solution:
    def reversePrint(self, head: ListNode) -> List[int]:
        out = []
        while head:
            out.append(head.val)
            head = head.next
        return out[::-1]
```

对于特殊情况，因为题目要求返回数组，所以即使head为空也要返回[]，这与out的初始化不谋而合，所以不用单独判断

**反转这种事都可以考虑用栈实现**

之所以没用是因为Python可以直接反转...而其他语言可能就需要最后用栈的方法实现

> 另解

**能用栈的事基本也能用递归实现**

想清楚递归公式和结束判断就好

+ 公式：[前] = [后] + [前.val] **python中list可以加法**
+ 结束：当head为空时不再递归，返回[]

```python
class Solution:
    def reversePrint(self, head: ListNode) -> List[int]:
        return (self.reversePrint(head.next) + [head.val]) if head else []
```

**需要注意python在函数中调用自己需要使用self**



#### [剑指 Offer 09. 用两个栈实现队列](https://leetcode-cn.com/problems/yong-liang-ge-zhan-shi-xian-dui-lie-lcof/)

> 自己思路

题目已经给定了一个队列与栈1，栈2

首先，当还没有出队操作时，只能是入栈1（要保证有序就只能入一个栈）

当出现出队操作时，利用栈2，将栈1中的序列负负得正

此时，若再入队，肯定还是进栈1（因为要负负得正），出队则出栈2

特殊情况是栈2空了，那就当从头开始再来一遍

```python
class CQueue:
    def __init__(self):
        self.s1 = []
        self.s2 = []

    def appendTail(self, value: int) -> None:
        self.s1.append(value)

    def deleteHead(self) -> int:
        if not self.s2:
            if not self.s1:
                return -1
            while self.s1:
                self.s2.append(self.s1.pop())
        out = self.s2.pop()
        return out
```

另外，也可以用两个队列实现栈，基本思路队列1存储，当需要出时从队列1中顺序出队，前面的都入队2，最后一个出掉

执行一次操作后，将队2队1互换，以此类推



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
def findRepeatNumber(self, nums: List[int]) -> int:    i = 0    while i < len(nums):        if nums[i] == i:            i += 1            continue            if nums[nums[i]] == nums[i]:                return nums[i]            nums[nums[i]], nums[i] = nums[i], nums[nums[i]]
```

> 附：注意赋值语句前后有相同元素时的修改顺序(最后一行的顺序反过来会报错)



> 附题：**不修改数组找出重复数字**

注意题干，若果是需修改数组找出重复数字，数组长度应比数字范围多1

>  思路：

对允许的数字范围进行二分，并计算该范围内数字在原数组出现次数

若出现次数多于范围长度，则说明有重复数字在此范围中，继续二分查找

```python
def getcount(left, right):            count = 0            for n in nums:                if left <= n <= right:                    count += 1            return count        start, end = 0, len(nums)-1        while start<=end:            mid = (start+end) // 2             count =  getcount(start, mid)            if start == end:                if count > 1:                    return start                else:                    return             if count > mid-start+1:                end = mid            else:                start = mid+1        return 
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
def findNumberIn2DArray(self, matrix: List[List[int]], target: int) -> bool:    if not matrix:        return False    w, h = len(matrix[0]), len(matrix)    i, j = 0, w-1    while i<h and j>=0:        if matrix[i][j] == target:            return True        if matrix[i][j] > target:            j -= 1        else:            i += 1    return False
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
def minArray(self, numbers: List[int]) -> int:    left, right = 0, len(numbers)-1    while left<right:        mark = numbers[right]        mid = (left+right)//2        if numbers[mid] > mark:            left = mid+1        elif numbers[mid] < mark:            right = mid        else:            # right -= 1            return min(numbers[left:right])    return numbers[left]
```



#### [剑指 Offer 50. 第一个只出现一次的字符](https://leetcode-cn.com/problems/di-yi-ge-zhi-chu-xian-yi-ci-de-zi-fu-lcof/)

> 自己思路：

没啥可说的，字典哈希表就完了

> 优化思路：

代码能更好看

```python
dic = {}for c in s:	dic[c] = not c in dicfor c in s:    if dic[c]:     	return c    return " "
```



#### [剑指 Offer 53 - I. 在排序数组中查找数字 I](https://leetcode-cn.com/problems/zai-pai-xu-shu-zu-zhong-cha-zhao-shu-zi-lcof/)

> 自己思路：

分两次分别查找target的左右两个端点(外边那个点)

将二分查找法写两次就完事

```python
def search(self, nums: List[int], target: int) -> int:	left, right = 0, len(nums)-1	# 这里使用<=循环找左端点，后边取值应取right(因为找的点比target小，所哟循环结束后right<left)    while left <= right:         mid = (left+right)//2        if nums[mid] >= target:        	right = mid-1        else:        	left = mid+1    start = right    left, right = 0, len(nums)-1    while left <= right:    	mid = (left+right)//2    	if nums[mid] <= target:        	left = mid+1        else:        	right = mid-1    # 这里同理    end = left    return end-start-1
```



> 优化思路：

写两次二分查找肯定存在冗余，提取出来写成函数

函数为提取右端点，左端点使用target-1查找，这样势必会找到target最左端，所以最后不用-1

```python
def search(self, nums: [int], target: int) -> int:    def helper(tar):        i, j = 0, len(nums) - 1        while i <= j:            m = (i + j) // 2            if nums[m] <= tar: i = m + 1            else:                 j = m - 1        return i    return helper(target) - helper(target - 1)
```



#### [剑指 Offer 53 - II. 0～n-1中缺失的数字](https://leetcode-cn.com/problems/que-shi-de-shu-zi-lcof/)

> 自己思路：

二分查找，更新left的方式换为序号与值相匹配

```python
def missingNumber(self, nums: List[int]) -> int:    i, j = 0, len(nums) - 1    while i <= j:        m = (i + j) // 2        if nums[m] == m:             i = m + 1        else:             j = m - 1    return i
```





### **双指针**

***

#### [剑指 Offer 18. 删除链表的节点](https://leetcode-cn.com/problems/shan-chu-lian-biao-de-jie-dian-lcof/)

> 自己思路：

链表删除节点需要记录前一个和当前两个节点

使用双指针，一个记录pre，一个记录cur

```python
def deleteNode(self, head: ListNode, val: int) -> ListNode:	if head.val == val:        return head.next    pre, cur = head, head.next    while cur:        if cur and cur.val == val:            pre.next = cur.next            break        pre = cur        cur = cur.next    return head
```

> 优化思路：

将判断条件放进循环，跳出循环直接执行删除节点操作

```python
def deleteNode(self, head: ListNode, val: int) -> ListNode:	if head.val == val:        return head.next    pre, cur = head, head.next    while cur and cur.val != val:        pre = cur        cur = cur.next    if cur:        pre.next = cur.next    return head
```



#### [剑指 Offer 21. 调整数组顺序使奇数位于偶数前面](https://leetcode-cn.com/problems/diao-zheng-shu-zu-shun-xu-shi-qi-shu-wei-yu-ou-shu-qian-mian-lcof/)

> 自己思路：

双指针，从前往后搜索偶数，从后往前搜索奇数，每次搜索到了互换

```python
def exchange(self, nums: List[int]) -> List[int]:    left, right = 0, len(nums)-1    while left < right:        while left<right and nums[left]%2 == 1:            left += 1        while left<right and nums[right]%2 == 0:            right -= 1        nums[left], nums[right] = nums[right], nums[left]    return nums
```

> 优化思路：

可以用&1替换除余判断奇偶数

原理是：&1会自动补全为&00001，相当于只计算最后一位&1，而二进制区分奇偶数正在于最后一位是0还是1

```python
def exchange(self, nums: List[int]) -> List[int]:    left, right = 0, len(nums)-1    while left < right:        while left<right and nums[left]&1 == 1:            left += 1        while left<right and nums[right]&1 == 0:            right -= 1        nums[left], nums[right] = nums[right], nums[left]    return nums
```

当然，考虑到对功能的抽象，可以把判断条件单独写为函数

```python
def isEven(num):	return num & 1 == 0    left, right = 0, len(nums)-1    while left < right:    	while left<right and not isEven(nums[left]):            left += 1        while left<right and isEven(nums[right]):            right -= 1        nums[left], nums[right] = nums[right], nums[left]    return nums
```



#### [剑指 Offer 22. 链表中倒数第k个节点](https://leetcode-cn.com/problems/lian-biao-zhong-dao-shu-di-kge-jie-dian-lcof/)

> 自己思路：

先遍历一遍找到链表的长度，再遍历找到输出位置

```python
def getKthFromEnd(self, head: ListNode, k: int) -> ListNode:    node, len = head, 1    while node.next:        len += 1        node = node.next    for _ in range(len-k):        head = head.next    return head
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

***

### **位运算**

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

***

### **数学**

#### [剑指 Offer 14- I. 剪绳子](https://leetcode-cn.com/problems/jian-sheng-zi-lcof/)

> 自己思路：

通过观察得知，越多划3结果越好，于是使用贪心算法，尽可能多的取3，剩下的部分分情况讨论

提交之后发现n<4都是特殊情况...得单独讨论，好麻烦啊，时间复杂度O(N)

```python
def cuttingRope(self, n: int) -> int:    if n<3:        return 1    if n==3:        return 2    out = 1    while n > 4:        out *= 3        n -= 3    return max(1,n)*out if n!=1 else n*4/3
```

于是又想到可以用递推的方法，把特殊情况先写进数组里，如下：

```python
def cuttingRope(self, n: int) -> int:	out = [0,0,1,2,4,6,9]    i = 6    while i<=n:        i += 1        out.append(out[i-3]*3)    return out[n]
```

复杂度没变但干净多了，时间复杂度O(N)

递推也可以写成动态规划，复杂度一样，把递推封装成函数就行啦，就不写了



> 优化思路：

时间复杂度O(1)

```python
def cuttingRope(self, n: int) -> int:	if n <= 3:         return n-1    a, b = n // 3, n % 3    if b == 0:         return int(3**a)    if b == 1:         return int(3**(a-1) * 4)    return int(3**a * 2)
```



#### [剑指 Offer 14- II. 剪绳子 II](https://leetcode-cn.com/problems/jian-sheng-zi-ii-lcof/)

> 自己思路：

与上题除了数的范围变大需要取余，没别的区别

对于python来说没有越界的事...直接取余就完了...不写了



#### [剑指 Offer 39. 数组中出现次数超过一半的数字](https://leetcode-cn.com/problems/shu-zu-zhong-chu-xian-ci-shu-chao-guo-yi-ban-de-shu-zi-lcof/)

> 自己思路：

排序找中位数O(NlogN)



> 优化思路：

摩尔投票法

本质思想就是对冲，找一对不同的数消去，重复这件事直到只剩下一个数，时间复杂度O(N)

```python
def majorityElement(self, nums: List[int]) -> int:    out, mark = nums[0], 0    for i in nums:        if i == out:            mark += 1        else:            mark -= 1        if not mark:            out = i            mark = 1    return out
```



#### [剑指 Offer 43. 1～n 整数中 1 出现的次数](https://leetcode-cn.com/problems/1nzheng-shu-zhong-1chu-xian-de-ci-shu-lcof/)

> 自己没思路

> 题解

统计所有1的个数不仅可以按数字顺序统计，也可以按位统计，统计每个位上1出现的次数

将数字按位分解为high,cur,low三部分，以cur为中心从低位至高位统计1的个数

假设数字为 HHHCLLL

统计C位1出现的次数，则有：

+ （与H相关，即HHH0000以下C位1的个数）：HHH * 10000
+ （与L相关，即HHH0000以上C位1的个数）：需要视C本位与1的关系分情况讨论

综上，需要引入的变量有H，C，L，计数用OUT，计当前位用WEI

时间复杂度O(lnN)代码如下：

```python
def countDigitOne(self, n: int) -> int:    wei, out = 1, 0    high, cur, low = n//10, n%10, 0    # high为0时cur还要再算一次，cur为0时很可能high没到头    while high or cur:        if cur == 0:            out += high * wei        elif cur == 1:            # 1xxx共有xxx+1个(1000)            out += high * wei + low + 1        else:            # 全覆盖就相当于high+1            out += (high+1) * wei         # 升位        high, cur, low, wei = high//10, high%10, cur*wei+low, wei*10    return out
```



#### [剑指 Offer 44. 数字序列中某一位的数字](https://leetcode-cn.com/problems/shu-zi-xu-lie-zhong-mou-yi-wei-de-shu-zi-lcof/)

> 自己没思路

> 题解：

为了求n位的数字，需要找到：

+ n位所在的序列中的数字是多少
+ n位在该数字中是第几位

然后就是喜闻乐见的找规律时间：

+ 去掉0，长度为n的数字共有9*10^(n-1)个

+ 由此可推，去掉0，长为n的数字共占n\*9\*10^(n-1)位

```python
def findNthDigit(self, n: int) -> int:    # 先找到区间    count, wei = 9, 1    # 先把位数小的去掉    while n>count:        n -= count        wei += 1        count = wei*10**(wei-1)*9    # 因为没算0，所以位数是多算了一位的    n -= 1    a, b = n//wei, n%wei    out = 10**(wei-1) + a    return int(str(out)[b])
```



#### [剑指 Offer 57 - II. 和为s的连续正数序列](https://leetcode-cn.com/problems/he-wei-sde-lian-xu-zheng-shu-xu-lie-lcof/)

> 自己思路：

列方程解方程组，注意输出的序列不要0，所以i从1开始遍历

```
def findContinuousSequence(self, target: int) -> List[List[int]]:	out = []	for i in range(1,target//2+1):		j = (2*target+i**2-i+1/4)**0.5-1/2		if j == int(j):			out.append(list(range(i, int(j) + 1)))	return out
```



> 另解

双指针

每次遇到这种问题都会觉得这种方法不能覆盖全（算是一种状态转移的方法）

换种方式想，**其实相当于让查询的数列徘徊在target周围**

双指针从小到大遍历，小于target就增加右指针，大于target就增加左指针

```python
def findContinuousSequence(self, target: int) -> List[List[int]]:	i, j, out = 1, 2, []	while i < j:		t = (i+j) * (j-i+1) / 2		if t == target:			out.append(list(range(i, j+1)))        if t <= target:            j += 1        else:            i += 1	return out
```



#### [剑指 Offer 62. 圆圈中最后剩下的数字](https://leetcode-cn.com/problems/yuan-quan-zhong-zui-hou-sheng-xia-de-shu-zi-lcof/)

> 自己思路：

最适合圆圈的数据结构应该是链表了吧，自己实现一个循环链表

> 优化思路：

f(n,m)执行第一次操作之后，第m个数字被删除

设f(5,3)，n=[1,2,3,4,5]，则f(n,m)的第二次操作等价于n=[4,5,1,2]进行f(4,3)的第一次操作

如果知道f(4,3)的结果，那么将结果+3(1 -> 4)再%4就得到了f(5,3)的结果

而我们恰好知道，当m=1时别无选择，f(1,m)=0，于是便可以进行递推了

```python
def lastRemaining(self, n: int, m: int) -> int:    out = 0    # i为当前数组长度    for i in range(2, n+1):        out = (out+m)%i    return out
```



#### [剑指 Offer 66. 构建乘积数组](https://leetcode-cn.com/problems/gou-jian-cheng-ji-shu-zu-lcof/)

> 自己思路：

暴力法，超时

> 优化思路：

O(N^2)超时的话只能考虑O(N)的方法，很显然乘法中有很大一部分重复

将问题考虑为一个矩阵，对角线元素为1时，整行相乘即为所求

为了优化计算量，可以将矩阵分解为上三角、下三角两个部分，每部分内可以使用推导公式

```python
def constructArr(self, a: List[int]) -> List[int]:	l, out = len(a), [1] * l	for i in range(1,l):		out[i] = out[i-1] * a[i-1]	tem = 1    for i in range(l-2,-1,-1):        tem *= a[i+1]        out[i] *= tem    return out
```



***

### **模拟**

#### [剑指 Offer 29. 顺时针打印矩阵](https://leetcode-cn.com/problems/shun-shi-zhen-da-yin-ju-zhen-lcof/)

> 自己思路：

按顺序换方向遍历，设置左右上下边界约束

```python
def spiralOrder(self, matrix: List[List[int]]) -> List[int]:	x, y = 0, 0	l, r, t, b = 0, len(matrix[0])-1, 0, len(matrix)-1	out = []	while True:		for i in range(l, r+1):            out.append(matrix[t][i])            t += 1        if t > b:        	break        for i in range(t, b+1):            out.append(matrix[r][i])            r -= 1        if l > r:        	break        for i in range(r, l-1, -1):            out.append(matrix[b][i])            b -= 1        if t > b:        	break        for i in range(b, t-1, -1):            out.append(matrix[i][l])            l += 1        if l > r:        	break	return out
```



#### [剑指 Offer 31. 栈的压入、弹出序列](https://leetcode-cn.com/problems/zhan-de-ya-ru-dan-chu-xu-lie-lcof/)

> 自己思路

模拟栈的操作过程就完了

循环入栈，每次入栈完判断栈中最后一个是否要出栈直到确认不出为止

```python
def validateStackSequences(self, pushed: List[int], popped: List[int]) -> bool:	stack, i = [], 0	for n in pushed:		stack.append(n)	while stack and stack[-1] == popped[i]:		stack.pop()		i += 1	return not stack
```



***

## **other**

