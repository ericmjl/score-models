---
title: A Pedagogical Introduction to Score Models
---

One of the coolest new developments in generative modelling
is the use of _score models_ to generate samples
that look like an existing collection of samples.
Pioneering this work is Yang Song,
who did this work while a PhD student at Stanford University.
The way I first uncovered his work is through Twitter,
where I was brought to his excellently written [blog post][yangblog] on the topic.

[yangblog]: https://yang-song.github.io/blog/2021/score/

In this collection of notebooks,
I would like to explore the fundamental ideas that underlie his work.
Along the way, we will work towards
a pedagogical implementation of score models as generative models.
By the end of this journey,
we should have a much better understanding of score models and their core concepts,
and should also have a framework for writing the code necessary
to implement score models in JAX and Python.
