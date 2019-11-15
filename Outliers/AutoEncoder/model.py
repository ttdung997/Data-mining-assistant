from lib.http_detection import HttpPredict
import redis
import threading
import time

lock = threading.Lock()

# Log error msg to log file at /opt/ModelApi/modelError.log
def log(msg):
	with open("/tmp/modelError.log", "a") as f:
		f.write("[%s]: %s\n" % (time.asctime(), msg))


# Pass request raw data to model and return result
def passModel(model, rawData):
	if rawData is None:
		log("Get data from redis return None!")
		return "1"

	prepareData = model.preprocess(rawData)
	if prepareData is None:
		log("Model preprocess return None!")
		return "1"

	with model.graph.as_default():
		result = model.predict(prepareData)
		# print(result)
	if result is None:
		log("Model predict return None!")
		return "1"

	if result < 0.5 :
		return "0"
	return "1"


# Get each request from redis queue, process and push back to redis
# Processed request will have 3 seconds timeout after expire
def process(model):
	while True:
		key = r.blpop("myRequestQueue")[1]
		if key is None:
			print("None key")
			continue
		rawData = r.get(key)
		result = passModel(model, rawData)

		r.set(key, result, ex=3)


if __name__ == '__main__':
	r = redis.StrictRedis(host='localhost', port=6379, db=0)
	if r is None:
		print("Cannot connect to redis server")
		exit()

	model = HttpPredict()
	model.loadModelInit()

	for i in range(4):
		procThread = threading.Thread(target=process, args=(model,))
		procThread.start()
