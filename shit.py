import confuse

def a(**kwargs):
	print(kwargs)
	print(aa)
	print(bb)
	dd = b(kwargs)
	

def b(cc):
	print(cc)


config = confuse.Configuration('Fluo_test', __name__)
config.set_file('config.yaml')

a(config)