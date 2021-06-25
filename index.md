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

### **位运算**

***

### **数学**

***

### **模拟**

***

## **other**

