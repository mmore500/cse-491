---
layout: post
title: "Concepts for higher-kinded types in C++20"
date: 2020-02-24
author: Brian Rago
---

# Concepts for Higher-Kinded Types in C++20

## Concepts

Later this year, the C++20 standard will be published. With C++20 come a great deal of new language features, including modules, ranges, `consteval`, and the spaceship operator. The feature that most captured my attention when I first heard about the new standard, however, was concepts.

A concept is a named set of requirements for template arguments. For an example of why this could be useful, consider the following:

```cpp
template<typename In, typename Out>
void copy(In in, Out out) {
    std::copy(std::cbegin(in), std::cend(in), std::begin(out));
}
```

This just hides the iterators in a call to `std::copy`. Could be a handy little procedure, since explicitly dealing with iterators can be ugly. Works just fine—when `In` and `Out` have a beginning and an end, that is. If not, then the compiler issues quite an error. 300 lines of mostly unintellegible garbage in my test, as the compiler attempted every implicit conversion it know from `int` in order to get `std::cbegin(int)` to compile. No luck. [Godbolt](https://godbolt.org/z/xCzVnL)

What if there were a way to tell the compiler that `copy` works only for a select few types? We can write a rule for what types are valid input ranges as a concept:

```cpp
template<typename T>
concept InputRange = requires(T a) {
    std::cbegin(a);
    std::cend(a);
};
```

A type `T` satisfies this concept (so `InputRange<T>` will be true) if and only if the two lines in the requires-expression compile. If we define a similar concept for output ranges, `copy` from before becomes:

```cpp
template<typename In, typename Out>
requires InputRange<In> && OutputRange<Out>
void copy(In in, Out out) {
    std::copy(std::cbegin(in), std::cend(in), std::begin(out));
}
```

Or better yet:

```cpp
template<InputRange In, OutputRange Out>
void copy(In in, Out out) {
    std::copy(std::cbegin(in), std::cend(in), std::begin(out));
}
```
[Godbolt](https://godbolt.org/z/BCjSPQ)

With this form, when I attempt to pass an integer as the first argument to `copy`, the compiler tells me right away exactly what's wrong:

```
<source>:17:6: note: constraints not satisfied
<source>: In instantiation of 'void copy(In, Out) [with In = int; Out = std::vector<int>]':
<source>:28:19:   required from here
<source>:6:9:   required for the satisfaction of 'InputRange<In>' [with In = int]
```

Very nice. Really, concepts are just a replacement for [SFINAE](https://en.cppreference.com/w/cpp/language/sfinae) and don't add any new capability to the language, but they certainly do improve error messages.

## Typeclasses

I'm a big fan of [Haskell](https://haskell.org). Haskell is a declarative, strongly-typed language. Each value has exactly one type. Haskell has first-class functions and a concise way to represent their types: a function that takes type `a` as an argument and provides type `b` as a result has type `a -> b`. We run into an issue, however, when we want to overload functions—what type should `(+)` have? `(+) :: Int -> Int -> Int` would mean we need to define another function with a distinct name in order to add values of type `Double`! Either that, or give addition the type `(+) :: a -> a -> a`, which simply does not make sense for non-numeric types.

Haskell's workaround is typeclasses, which had never been seen built into a language before Haskell 1.0 in 1990. Typeclasses work with Haskell's [parametric polymorphism](https://en.wikipedia.org/wiki/Parametric_polymorphism) to allow us to restrict the types that can act as `a` in that last signature for `(+)`. A typeclass is defined as follows:

```haskell
class Add a where
    (+) :: a -> a -> a
```

Now, the function `(+)` has type `Add a => a -> a -> a` (read "`a -> a -> a`, where `a` is of class `Add`"). Providing typeclass instances for `Add Int` and `Add Double` is done like so:

```haskell
instance Add Int where
    x + y = {- something -}

instance Add Double where
    x + y = {- something -}
```

Now both can be added with the function `(+)`, and we still need not define a nonsensical overload for non-numeric types. Good stuff.

One notable typeclass in Haskell's standard library is `Monoid`, representing a type with an associative binary operator `mappend` and an identity element `mempty` and defined as follows:

```haskell
class Monoid a where
    mempty :: a
    mappend :: a -> a -> a
```

Note here that `mempty` has an interesting type mostly alien to popular languages: `Monoid a => a`. It's an overloaded value, almost like `template<typename T> T mempty` in C++. Whenever some type `a` is a Monoid, `mempty` can be of type `a`.

## Putting it together

A concept is a set of requirements for template arguments, which can be types. A typeclass is a set of operations that can be performed on types. Very similar, huh? My first thought when learning of concepts was that I should try to implement Haskell typeclasses in C++. My first attempt:

```cpp
template<typename T> T mempty;
template<typename T> T mappend(T, T);

template<typename T>
concept Monoid = requires(T a) {
    { mempty<T> } -> T;
    { mappend<T>(a, a) } -> T;
};
```

This seemed at first to work okay:
```cpp
template<> int mempty<int> = 0;
template<> int mappend<int>(int a, int b) { return a + b; }
```

But I soon ran into an issue when trying to write a Monoid instance for my own linked-list type (identity is empty, and append is concatenation):
```cpp
template<typename T> List<T> mempty<List<T>> = null<T>;
```
Wait a minute. This isn't right. A full template specialization should have no template arguments, not one, but there is no way to declare `typename T` except in the template argument list. This was my first glimpse of the struggles to come, but I did not yet pick up on what would eventually be my takeaway from this venture.

If C++ had template templates, this approach would be very possible, like so:
```cpp
template<>              // Full specialization
template<typename T>    // For all types T...
List<T> mempty<List<T>> = null<T>;
```
I do not yet know enough of C++ to understand why this is not allowed.

[Nitash](https://github.com/cgnitash) provided the following approach, similar but perhaps somewhat improved (actually working, at least):

```cpp
template<typename T> T mappend(T, T) = delete;

template <> int mappend(int, int);
template<typename T> List<T> mappend(List<T>, List<T>);

template<typename T>
concept Monoid = requires (T a) {
    { T{} } -> T;
    { mappend(a, a) } -> T; 
};
```

Variable templates can't be deleted, so we have to settle for asserting that brace-initialized `{}` will always result in `mempty`. Compared to defining a function template that takes no parameters and returns some value, this approach has the advantage of working (the other can't express two Monoid instances for `List<T>` and `std::vector<T>`, for example), though not very well. The `Bounded` typeclass in Haskell has two values, `minBound` and `maxBound`, both of type `Bounded a => a`—these can't both be expressed with `{}`!

Some struggling with templates later, I gave up on making a satisfactory Monoid typeclass and moved on to higher-kinded types, the indended topic of this post.

## Kind

A _kind_ is a type of types. A type that can be instantiated is a _concrete type_, we say, and a concrete type's kind is represented by `*`. Examples include `int`, `std::optional<float>`, and `std::pair<std::vector<std::string>, bool>` in C++.

Some types cannot be instantiated. Abstract classes in C++, but let's ignore those. OOP has no place in my attempt at elegant FP. I'm talking about class templates. [`std::vector` is an example.](https://godbolt.org/z/sND-Ly) What inner type would a plain vector hold? You can think of `std::vector` as a function on types, taking a concrete type `T` to the concrete type `std::vector<T>`. It has kind `* -> *`.

A higher-kinded type is a non-concrete type. Some examples are `std::vector :: * -> *` and `std::pair :: * -> * -> *` (and their Haskell approximations, `[]` and `(,)`). Concrete or not, these are types nevertheless, and can therefore belong to typeclasses. Not the same typeclasses, of course—all I've yet mentioned are classes of concrete types. The simplest interesting class I know of higher-kinded types is Functor:

```haskell
class Functor (f :: * -> *) where
    fmap :: (a -> b) -> f a -> f b
```

A Functor `f` is a type of kind `* -> *` that allows one to map a function over the values it contains. There are some laws for the behavior of `fmap`, but I'll skip over them. The List instance can be approximated in C++ using `std::vector` as follows:
```cpp
template<typename A, typename Fn, typename B = std::invoke_result<Fn, A>>
std::vector<B> fmap(Fn f, std::vector<A> as) {
    std::vector<B> bs(as.size());
    std::transform(std::cbegin(as), std::cend(as), std::begin(bs), f);
    return bs;
}
```

Other simple Functors include `std::optional` (`Maybe` in Haskell) and `std::pair<A>` (if partial application of templates were allowed). Exercise: define `fmap` for each of these. A more surprising Functor is `(->) x`, the type of functions taking type `x` as an argument. Consider:
```haskell
p :: a -> b    -- A function from x to y
q :: x -> a    -- Where out functor f is functions from x, q is of type f a

fmapped x = p (q x)
-- One way to combine p and q.
-- fmapped is a function taking an argument of type x
-- Apply q (of type x -> a) to x, yielding a value of type a
-- Apply p to this value, yielding a value of type b
-- So fmapped has type x -> b, or f b
```
In the case of functions, `fmap` is composition. Pretty cool.

## Putting it together (reprise)

Okay, time to define Functors in C++. First attempt:
```cpp
template<template<typename> typename F>       // F has kind * -> *
concept Functor = requires(F<?> fa, ? f) {    // Need to declare the inner type and the function's type somehow
    { fmap(f, fa) } -> F<?>;    // Also the resulting type.
};
```

Unlike Haskell, C++ does not allow us to use type variables without first declaring them, and this declaration must come in a very particular place. Second attempt:
```cpp
template<template<typename> typename F, typename A, typename Fn, typename B = std::invoke_result<Fn, A>>
concept Functor = requires(F<A> fa, Fn f) {
    { fmap(f, fa) } -> F<B>;
};
```

Ooh, a concept that [compiles](https://godbolt.org/z/gSer_7)! Does it do what I want? Unfortunately, it does not. Very close, yes, and it will likely work (at least somewhat) for what I want to do, but I am not interested in continuing with this fundamentally flawed typeclass.

What's wrong with it? Recall that Functor is a class of higher-kinded types `f :: * -> *` admitting a suitable `fmap :: (a -> b) -> f a -> f b`. Look at the template arguments in the concept—what I've defined here is a class of sets of four types! Whether or not it behaves as a Functor class should behave, all meaning once attached to the word is now lost. I find this unacceptable.

Third attempt:
```cpp
template<template<typename> typename F>
template<typename A, typename Fn, typename B = std::invoke_result<Fn, A>>
concept Functor = requires(F<A> fa, Fn f) {
    { fmap(f, fa) } -> F<B>;
};
```

Again, a template template. Again, not legal in C++. Fourth attempt:

```cpp
template<template<typename> typename F>
concept Functor = template<typename A, typename Fn, typename B = std::invoke_result<Fn, A>>
                  requires(F<A> fa, Fn f) {
    { fmap(f, fa) } -> F<B>;
};
```

Okay, now I'm grasping at straws. I really want to find something that works, and that has yet to happen. A requires template is not part of the C++ grammar, and it looks hideous, but it would remedy my woes. [I turned to Stack Overflow](https://stackoverflow.com/q/60547534/5499914) regarding this problem.

## Takeaways

My question was graced with [an excellent answer](https://stackoverflow.com/a/60549046/5499914) by user [Barry](https://stackoverflow.com/users/2069064/barry). This answer begins:

> C++ doesn't have parametric polymorphism like this - you can't do things like "for any type" in the way you want to, and in the way you can in Haskell. I think that's fundamentally impossible in a world where overloading exists.

Fundamentally impossible, huh? Is C++ simply <del>not expressive enough</del> expressive in the wrong way to express such an idea? Always on the lookout for things I can claim make Haskell a better language than C++, I just had to try to come up with some reasoning as to why this might be the case.

Pretty clear after a little bit of thought. This concept approach to typeclasses does not bind methods (as they're sometimes called in Haskell) to the concept at all. All these "methods" must be free functions declared before the concept is introduced; the concept is nothing more than a check for their existence. One of these functions could easily be overloaded like so:
```cpp
template<template<typename> typename F, typename Fn>
int fmap<F, Fn, int, int>(Fn f, F<int> xs) { return 0; }
```
Uh-oh. If `Functor` is defined only in terms of one higher-kinded type, then the mere existence of this overload means our typeclass must fail for all such types, since `fmap` doesn't return the correct type when mapping `int` to `int`.

In short, what I want to do is impossible.

There are other problems with concepts as typeclasses. Haskell provides the following instance:
```haskell
instance Monoid a => Monoid (Maybe a) where    -- "When a is a Monoid, so is Maybe a"
    mempty = Nothing

    mappend Nothing b = b
    mappend a Nothing = a
    mappend (Just a) (Just b) = Just (mappend a b)
```
To use this with my C++ approach would require declaring the concept, then providing the instance for `std::optional`, then defining the concept. Concepts cannot be forward-declared, so this as well is impossible.

Overall, I'm quite disappointed with concepts. I wanted to use them to port [Parsec](https://hackage.haskell.org/package/parsec) to C++, but that's tabled indefinitely until I can figure out a sensible way to deal with monads.
