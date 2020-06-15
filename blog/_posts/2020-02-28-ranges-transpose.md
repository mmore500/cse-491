---
layout: post
title: "Matrix Multiplication With Range-v3"
date: 2020-04-20
author: Bradley Bauer
---

In this post we will work through a few examples of using range-v3.
I'll be using range-v3 because C++20's std::ranges does not include all the view adaptors used in these examples. (Specifically, chunk and stride are not in C++20 according to cppreference.)

All the code in this post is on [godbolt](https://godbolt.org/z/uMmq8f) if you'd like to take a closer look. If you want to compile on your own machine then you can get started with
```
git clone https://github.com/xdaimon/ranges_blog
cd ranges_blog
git clone https://github.com/ericniebler/range-v3
./build.sh && ./a.out
```
Throughout I will assume the following header has been included.
```cpp
#include <bits/stdc++.h>
#include <range/v3/all.hpp>
namespace rs = ranges;
namespace vs = ranges::views;
auto print=[](auto rng){std::cout<<rng<<std::endl;};
auto print2D=[&](auto rng){for(auto r:rng)print(r);};
```

# A Brief Introduction to Ranges

There is already a [bunch](https://hannes.hauswedell.net/post/2019/11/30/range_intro/) of [great](https://www.fluentcpp.com/2018/02/09/introduction-ranges-library/) [introductions](https://www.modernescpp.com/index.php/c-20-the-ranges-library) to ranges so I will give just a brief overview here. First some vocab,
  * <b>range</b>: a collection of things that can be iterated over.
  * <b>view</b>: like a transformed image of some other range. Views are cheap to copy since they do not store their underlying range. A view is a range and so it is possible to form a view over another view.
  * <b>view adaptor</b>: something that takes a range and produces a view of that range. View adaptors use `operator|` to perform this transformation: `viewOfRange = someRange | adaptor;`. 

One nice thing about views is that they can be composed easily. For example, instead of having to write
```cpp
auto tempView = someRange | adaptor1;
auto temp2View = tempView | adaptor2;
auto desiredView = temp2View | adaptor3;
```
we can simply write
```cpp
auto desiredView = someRange | adaptor1 | adaptor2 | adaptor3;
```

Finally, let's see some of the views used throughout the rest of the post.

```cpp
auto x = vs::ints(1,5+1);  // [1,2,3,4,5]
print(x | vs::drop(2));    // [3,4,5] drop the first two elements of x
print(x | vs::stride(2));  // [1,3,5] take every other element of x
print2D(x | vs::chunk(2)); // [[1,2], group elements of x into chunks of length two
                           //  [3,4],
                           //  [5]]
print(x | vs::chunk(2) | vs::join); // [1,2,3,4,5] join concatenates a range of ranges
print(x | vs::transform([](auto xi){ return 2*xi; })); // transform maps a lambda over a range
                                                       // [2,4,6,8,10]
auto y = std::vector{1,2,3,4};
print(rs::inner_product(x, y, -.5)); // 29.5
print(rs::distance(y)); // 4
```

# Matrix Product and Transpose

As an example of view composition I'll implement the matrix product which will also require us to implement the matrix transpose. Let's use transform and `inner_product` to compute the matrix-vector product *Wx*.
```cpp
auto x = vs::ints(1,3+1); // [1,2,3]
auto W = vs::ints(1,2*3+1) | vs::chunk(3); // [1,2,3]
                                           // [4,5,6]
print(W | vs::transform([&](auto row){ return rs::inner_product(row, x, 0); })); // [14,32]
```

Computing the matrix product *XW*, with *X* and *W* each represented as a range of rows, is more involved.
One way to do it is to use `inner_product` between the rows of *X* and columns of *W*. This would be easy if we had a range over the columns of *W*.
But, to get such a range we need to compute the transpose of *W*.
```cpp
auto W = vs::ints(1,3*2+1) | vs::chunk(2); // [1,2]
                                           // [3,4]
                                           // [5,6]
print2D(vs::ints(0,2) | vs::transform([&](int i) { // for each column
  return W | vs::join // concatenate all the rows into a single range
           | vs::drop(i) // remove everything before the 1st element of the ith column
           | vs::stride(2); // take every Nth item to provide a view of the ith column
})); // [1,3,5]
     // [2,4,6]
```

Let's put that into a function.

```cpp
auto transpose = [](auto rng) {
  auto flat = rng | vs::join;
  int height = rs::distance(rng);
  int width = rs::distance(flat) / height;
  auto inner = [=](int i) {
    return flat | vs::drop(i) | vs::stride(width);
  };
  return vs::ints(0,width) | vs::transform(inner);
};
```
`inner` captures by value because it does not execute until after the `rng` variable has been destroyed. Unfortunately, this does not compile. Clang gives errors that suggest `flat`, as accessed from inside `inner`, does not satisfy the range concept. We can check this directly with a static assert.
```cpp
auto inner = [=](int i) {
  static_assert(rs::range<decltype(flat)>); // This static assert fails
  return flat | vs::drop(i) | vs::stride(width);
};
```
The problem is solved by marking `inner` as mutable.
```cpp
auto inner = [=](int i) mutable {
  return flat | vs::drop(i) | vs::stride(width);
};
```
Adding the mutable specification causes `inner` to capture as non-const (the default for capture by value is const). I think the compilation error had something to do with join returning an "input_range". An input_range is a type of range that can be iterated over *at least* once. If we had a const view over an input_range, and the range could be iterated over only a finite number of times, then how would the const view keep track of how many times it had been iterated over? It must keep some internal state and therefore cannot be declared const. This is not much more than a guess though since I have not dived too deep into the library implementation and the documentation is sparse.

After making this correction to `transpose`, we can easily implement matrix multiplication.
```cpp
auto X = vs::ints(1,2*3+1) | vs::chunk(3); // [1,2,3]
                                           // [4,5,6]
auto W = vs::ints(1,3*2+1) | vs::chunk(2); // [1,2]
                                           // [3,4]
                                           // [5,6]
print2D(X | vs::transform([&](auto xrow) {
  return transpose(W) | vs::transform([=](auto wcol) {
    return rs::inner_product(xrow, wcol, 0);
  }); // [22,28]
}));  // [49,64]
```

# Conclusions
One last note.
If you write code using range-v3 you may find that the compiler's error messages are difficult to understand.
One reason for this is because ranges-v3 emulates concepts through a mixture of macros and template metaprogramming.
So when things fail, the error messages have to do with things deep in the emulation's implementation.
Hopefully std::ranges will be able to make use of non-emulated concepts to fail in a more graceful way.

In sum, I think ranges are a very nice addition to the standard library. I hope these examples helped you get a feel for what can be done with ranges and how you might use ranges in your own code.

Here are a few links I found useful while learning about ranges.
  * [C++ code samples before and after Ranges](https://mariusbancila.ro/blog/2019/01/20/cpp-code-samples-before-and-after-ranges/)
  * [A beginner's guide to C++ Ranges and Views](https://hannes.hauswedell.net/post/2019/11/30/range_intro/)
  * [Tutorial: Writing your first view from scratch](https://hannes.hauswedell.net/post/2018/04/11/view1/)
  * [Introduction to the C++ Ranges Library](https://www.fluentcpp.com/2018/02/09/introduction-ranges-library/)
  * [The Surprising Limitations of C++ Ranges Beyond Trivial Cases](https://www.fluentcpp.com/2019/09/13/the-surprising-limitations-of-c-ranges-beyond-trivial-use-cases/)
  * [C++20: The Ranges Library](https://www.modernescpp.com/index.php/c-20-the-ranges-library)
  * [The Range-v3 User Manual](https://ericniebler.github.io/range-v3/)
  
## Comments? Questions?

Jump on the twitter thread below to chat!! ☎️ ☎️ ☎️

<blockquote class="twitter-tweet" data-conversation="none"><p lang="en" dir="ltr">Bradley Bauer wrote up some demos of matrix operations using the ranges-v3 library (which is moving into the stl with <a href="https://twitter.com/hashtag/cpp20?src=hash&amp;ref_src=twsrc%5Etfw">#cpp20</a><a href="https://t.co/D7d7NjLVRG">https://t.co/D7d7NjLVRG</a></p>&mdash; Matthew A Moreno (@MorenoMatthewA) <a href="https://twitter.com/MorenoMatthewA/status/1272329715261440000?ref_src=twsrc%5Etfw">June 15, 2020</a></blockquote> <script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>
