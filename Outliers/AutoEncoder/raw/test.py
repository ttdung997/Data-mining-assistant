normal = [line.rstrip('\n') for line in open('soict_normal.txt.reform')]
anomaly = [line.rstrip('\n') for line in open('soict_anomalous.txt.reform')]

for line in anomaly:
	if line not in normal:
		print(line)
