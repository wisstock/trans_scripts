#!/usr/bin/env python3

""" Copyright © 2020-2021 Borys Olifirov

Experiments with OOS composition

links:
https://webdevblog.ru/obyasnenie-classmethod-i-staticmethod-v-python/
https://webdevblog.ru/nasledovanie-i-kompoziciya-rukovodstvo-po-oop-python/
https://stackoverflow.com/questions/49370769/python-init-and-classmethod-do-they-have-to-have-the-same-number-of-args

"""


# import confuse

class aaa():
	def __init__(self):
		pass

	@classmethod
	def settings(cls, a=0):
		cls.a = a

	def add(self, c):
		self.c = c + self.a
		return self.c


class bbb():
	def __init__(self, b=2):
		self.b = b
		self.AAA = aaa()
		self.det = self.AAA.add(c=self.b)

	def plus(self):
		print(self.b, self.AAA.a)
		print(self.det)


aaa.settings(a=2)
var = bbb()
var.plus()