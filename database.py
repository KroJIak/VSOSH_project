import json

class dbWorker:
	def __init__(self, databaseFilePath):
		self.databaseFilePath = databaseFilePath

	def get(self):
		with open(self.databaseFilePath) as file:
			dbData = json.load(file)
		return dbData
