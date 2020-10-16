class Solution:
    def magicalString(self, n: int) -> int:

        i = 2
        magicS = "122"
        while (len(magicS) < n):
            if magicS[i] == 1:
                code = '2'
            else:
                code = '1'
            magicS += code * int(magicS[i])
            i += 1

        return magicS[:n].count('1')

ret = Solution().magicalString(6)
print(ret)