dataset  = "raw/cisc/data.csv"
data = get_data(dataset)

# Generate a dictionary of valid characters
valid_chars = {"":1," ":2,"!":3,"\"":4,"#":5,"$":6,"%":7,"&":8 ,"'":9 ,"(":10,")":11,"*":12 ,"+":13,",":14,"-":15,".":16,"/":17,"0":18,"1":19,"2":20,"3":21,"4":22,"5":23,"6":24,"7":25,"8":26,"9":27,":":28,";":29,"<":30,"=":31,">":32,"?":33,"@":34,"A":35,"B":36,"C":37,"D":38,"E":39,"F":40,"G":41,"H":42,"I":43,"J":44,"K":45,"L":46,"M":47,"N":48,"O":49,"P":50,"Q":51,"R":52,"S":53,"T":54,"U":55,"V":56,"W":57,"X":58,"Y":59,"Z":60,"[":61,"\\":62,"]":63,"^":64,"_":65,"`":66,"a":67,"b":68,"c":69,"d":70,"e":71,"f":72,"g":73,"h":74,"i":75,"j":76,"k":77,"l":78,"m":79,"n":80,"o":81,"p":82,"q":83,"r":84,"s":85,"t":86,"u":87,"v":88,"w":89,"x":90,"y":91,"z":92,"{":93,"|":94,"}":95,"~":96}
max_features = len(valid_chars)
list_chars = [x for x in range(96)]
encoded = to_categorical(list_chars)
i =0
for value in valid_chars:
    valid_chars[value] = encoded[i]
    i = i+1

maxlen = 74
X = [[valid_chars[y] for y in x] for x in data]
X = sequence.pad_sequences(X, maxlen=maxlen)
X_label = np.flip(X,1)

"""Build LSTM AutoEncoder model"""
model = Sequential()
# model.add(Embedding(max_features, 128, input_length=maxlen))
model.add(LSTM(64,return_sequences=True,go_backwards=True))
model.add(LSTM(max_features,return_sequences=True))
# # # model.add(Dropout(0.5))
# model.add(Dense())
# # # # model.add(Activation('softmax'))
# model.add(Lambda(lambda x: K.argmax(x)))
model.compile(loss='mean_squared_error',optimizer='adam')
model.fit(X, X_label, batch_size=32, epochs=1)