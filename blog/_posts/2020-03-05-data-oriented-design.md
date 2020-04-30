---
layout: post
title: "C++ Data-Oriented Design Paradigm"
date: 2020-03-05
author: Sean Kennedy
---

When software developers consider what “good” software design looks like, most minds instinctually wander toward some model of **object-oriented programming (OOP)**. 

We lean so heavily on this paradigm not only because most of us were taught this way, but also because an object-oriented approach—**structuring code based on objects in our perceived model of the real world**—is generally a very good way of designing software. **Encapsulation, abstraction, inheritance, and polymorphism** are four very important principles in software design that work nicely within the object-oriented paradigm while providing high levels of **code reusability** and **maintainability**. The merits of object-oriented programming are numerous and well documented, so much so that I am going to assume that the reader is already familiar with them on some level. 

While there is nothing inherently wrong with object-oriented design, **developers can end up programming themselves into a box because they have constrained themselves within the dimensions of an object-oriented world**. Code that is modeled well from an object-oriented perspective can actually disguise the fact that the code is not solving the problem at hand in the most logical way, undermining the purpose of writing the code in the first place. Taking an approach centered around how the code is really working to transform program data—**a data-oriented approach**—can help programmers avoid the pitfalls that object-oriented code can conceal.  

## An Introduction to Data-Oriented Design (DOD)
When taking a strictly object-oriented approach to writing software, we tend to lose sight of the goals underlying our code.

**Code is a step-by-step guide through which humans can explain to a particular computer system how to solve a given problem in an efficient way.** Some problems are obviously more complicated than others, but most all problems written in software boil down to the manipulation of data from one form into another. Code is merely the tool that programmers use to solve this data transformation problem. To solve this problem in the most optimal way, programmers must have an understanding of how their code affects the hardware platform that they are targeting.  This is the part that is ignored all too often by programmers in favor of prioritizing a specific model of code. 

**A common example of this is how programmers write their code without consideration for how their program is utilizing the CPU cache.** Caches allow a computer to take advantage of data locality to greatly increase memory-access efficiency; however, if the programmer does not write their program in consideration of data locality, they are missing out on this key performance boost.

For most applications, these performance advantages are relatively minor, but for problems where speed is of the essence, a data-oriented approach to problem solving is paramount. But what do these ideas look like in real code?

## Data-Oriented Design: Example in C++ 
This example looks at two different implementations of a simple stock portfolio management system in C++. The first code snippet shows a class structure consistent with **object-oriented design** principles. 
```cpp
struct StockPosition
{
    std::string stockSymbol;
    double stockValue;
    double stockAverageCost;
    unsigned int numShares;
};

class OopStockPortfolio
{
private:
    std::vector<StockPosition> stockPortfolio;
    unsigned int size;

public:
    void constructPortfolioFromFile(const std::string& fileName);
    double getPortfolioValue();
    double getPortfolioCost();
    double getTotalReturn();
};
```
The key data member in this class is the stock portfolio which consists of a vector of stock position structs. Each stock position consists of a stock symbol, the current value of a share of that stock, the number of shares owned, and the average cost paid for each stock. Modeling a stock portfolio as a list of stock position models is a very intuitive way to structure the code. **However, performing computations on this data structure requires looping through the stock portfolio and accessing the memory chunk for each stock position, even when the computation only needs a small piece of information about each stock position.**

Alternatively, a **data-oriented design** of a stock portfolio system might look something like this:  
```cpp
class DodStockPortfolio
{
private:
    std::vector<std::string> stockSymbols;
    std::vector<double> stockValues;
    std::vector<double> stockAverageCosts;
    std::vector<unsigned int> numShares;
    std::unordered_map<std::string, int> stockIndices;
    unsigned int size;
    unsigned int getStockIndex(const std::string& stockSymbol);

public:
    void constructPortfolioFromFile(const std::string& fileName);
    double getPortfolioCost();
    double getPortfolioValue();
    double getTotalReturn();
};

```
The key difference here is that rather than using an **array of structures (AoS)** like we had in the first example, this class uses a **structure of arrays (SoA)** to store all the data associated with the stock portfolio. This allows us to perform computations that iterate over arrays of smaller data members which can be more efficiently cached. **This can significantly boost the speed of the application in which this class is implemented.**

An implementation of computational methods in the **object-oriented** version might look like this:
```cpp
double OopStockPortfolio::getPortfolioValue()
{
    double returnVal = 0;
    std::for_each(stockPortfolio.begin(), stockPortfolio.end(), 
        [&returnVal](const StockPosition &a) { returnVal += (a.numShares * a.stockValue); });
    return returnVal;
}

double OopStockPortfolio::getPortfolioCost()
{
    double returnVal = 0;
    std::for_each(stockPortfolio.begin(), stockPortfolio.end(), 
        [&returnVal](const StockPosition &a) { returnVal += (a.numShares * a.stockAverageCost); });
    return returnVal;
}

double OopStockPortfolio::getTotalReturn()
{
    return (getPortfolioValue() / getPortfolioCost()) * 100 - 100.0;
}
```

Meanwhile, an implementation of computational methods in the **data-oriented** version might look like this:
```cpp
double DodStockPortfolio::getPortfolioValue()
{
    double returnVal = 0;
    for (int i = 0; i < size; i++)
    {
        returnVal += (numShares[i] * stockValues[i]);
    }
    return returnVal;
}

double DodStockPortfolio::getPortfolioCost()
{
    double returnVal = 0;
    for (int i = 0; i < size; i++)
    {
        returnVal += (numShares[i] * stockAverageCosts[i]);
    }
    return returnVal;
}

double DodStockPortfolio::getTotalReturn()
{
    return (getPortfolioValue() / getPortfolioCost()) * 100 - 100.0;
}
```
When the total return is computed on a stock portfolio consisting of 1500 stock positions, **the data-oriented implementation outperforms the object-oriented implementation by a factor of 2.5 on average** (GCC 7.5.0; Ubuntu 18.04 64-bit; g++ -O2). Even for such a small piece of software, the change in the structure of the code resulted in a significantly more efficient system because the hardware has more direct access to the data. Based on this example, it is easy to see how object-oriented code written without regard for the underlying hardware can result in an inferior system, even when the code structure seems sensible.  

## Can DOD and OOP coexist?
Yes, they not only *can* coexist; they *should* coexist. This is because data-oriented design and object-oriented programming are not mutually exclusive. **Data-oriented design is more like a programming philosophy rather than a coding framework.** It is characterized by programming with a primary focus on solving the data transformation problem before even considering code design. Once the programmer has decided on how the hardware should interact with the data, the code can then be designed around this data. 

From here, the programmer can begin to analyze the tradeoffs between different object-oriented models. Often, it is reasonable to sacrifice performance for a more maintainable and/or reusable codebase, but **the programmer can only understand the value of this tradeoff if they understand how it affects the relationship between the hardware and the data.** This can only be done if the programmer takes a data-oriented approach to writing their code. 

## Conclusion
When writing code with high-level programming languages, it is easy to get lost in the convenient levels of abstraction afforded to us, and this leads us to forget that our code is responsible for telling hardware how to transform data in a very specific way at a lower level. A data-oriented philosophy leads to code that is more fundamentally sound, and it forces developers to truly understand what their programming language is doing under the hood. This is the way software is meant to be written. 

Being conscientious of data-oriented design may or may not radically change the way you write code, but having a better understanding of why it is important will almost certainly make you a better developer. If you want to learn more, check out some of the resources below. Thanks for reading!

## Resources

- [CppCon 2014: Mike Acton "Data-Oriented Design and C++"](https://www.youtube.com/watch?v=rX0ItVEVjHc)
- [CppCon 2018: Stoyan Nikolov “OOP Is Dead, Long Live Data-oriented Design”](https://www.youtube.com/watch?v=yy8jQgmhbAU)
- [CppCon 2019: Matt Godbolt “Path Tracing Three Ways: A Study of C++ Style”](https://www.youtube.com/watch?v=HG6c4Kwbv4I)

