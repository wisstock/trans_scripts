

def a(aa, bb, **kwargs):
	print(kwargs)
	ff = b(**kwargs)
	return aa + bb + ff

def b(cc, dd):
	print(cc)
	print(dd)
	return cc + dd

print(a(1, 2, cc = 3, dd = 4))