---
title: Post template
description: Short summary of the post
date: 2024-11-30 18:32
categories: [Misc, Template]
tags: [tag1, tag2]     # TAG names should always be lowercase
math: true
pin: true
---

# How can I write a new post?

Refer to articles below:  
[https://chirpy.cotes.page/posts/write-a-new-post/](https://chirpy.cotes.page/posts/write-a-new-post/)  
[https://chirpy.cotes.page/posts/text-and-typography/](https://chirpy.cotes.page/posts/text-and-typography/)

# How can I add math equations?

<!-- Block math, keep all blank lines -->

$$
\frac{x}{y} = 0.05
$$

<!-- Equation numbering, keep all blank lines  -->

$$
\begin{equation}
  y = x + 1
  \label{eq:1}
  \tag{2.1.1}
\end{equation}
$$

Can be referenced as $ 2.1.1 $.

<!-- Inline math in lines, NO blank lines -->

"Lorem ipsum dolor sit amet, $$ y = 3 $$ consectetur adipiscing elit."

<!-- Inline math in lists, escape the first `$` -->

1. \$$ 1 + 1 = 2 $$
2. \$$ 2 + 2 = 4 $$
3. \$$ 3 + 3 = 6 $$

# How can I add a code block?

```C++
#include <bits/stdc++.h>
using namespace std;

int main(void) {
    ios::sync_with_stdio(0);
    cin.tie(0); cout.tie(0);
    cout << "hi";
    return 0;
}
```