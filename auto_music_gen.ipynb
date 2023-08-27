#DataFlair 
#load all the libraries
from music21 import *
import glob
from tqdm import tqdm
import numpy as np
import random
from tensorflow.keras.layers import LSTM,Dense,Input,Dropout
from tensorflow.keras.models import Sequential,Model,load_model 
from sklearn.model_selection import train_test_split

def read_files(file):
  notes=[]
  notes_to_parse=None
  #parse the midi file
  midi=converter.parse(file)
  #seperate all instruments from the file
  instrmt=instrument.partitionByInstrument(midi)

  for part in instrmt.parts:
    #fetch data only of Piano instrument
    if 'Piano' in str(part):
      notes_to_parse=part.recurse()

      #iterate over all the parts of sub stream elements
      #check if element's type is Note or chord
      #if it is chord split them into notes
      for element in notes_to_parse:
        if type(element)==note.Note:
          notes.append(str(element.pitch))
        elif type(element)==chord.Chord:
          notes.append('.'.join(str(n) for n in element.normalOrder))

  #return the list of notes
  return notes

#retrieve paths recursively from inside the directories/files
file_path=["schubert"]
all_files=glob.glob('All Midi Files/'+file_path[0]+'/*.mid',recursive=True)

#reading each midi file
notes_array = np.array([read_files(i) for i in tqdm(all_files,position=0,leave=True)])

#unique notes
notess = sum(notes_array,[]) 
unique_notes = list(set(notess))
print("Unique Notes:",len(unique_notes))

#notes with their frequency
freq=dict(map(lambda x: (x,notess.count(x)),unique_notes))

#get the threshold frequency
print("\nFrequency notes")
for i in range(30,100,20):
  print(i,":",len(list(filter(lambda x:x[1]>=i,freq.items()))))

#filter notes greater than threshold i.e. 50
freq_notes=dict(filter(lambda x:x[1]>=50,freq.items()))

#create new notes using the frequent notes
new_notes=[[i for i in j if i in freq_notes] for j in notes_array]

#dictionary having key as note index and value as note
ind2note=dict(enumerate(freq_notes))

#dictionary having key as note and value as note index
note2ind=dict(map(reversed,ind2note.items()))

#timestep
timesteps=50

#store values of input and output
x=[] ; y=[]

for i in new_notes:
  for j in range(0,len(i)-timesteps):
    #input will be the current index + timestep
    #output will be the next index after timestep
    inp=i[j:j+timesteps] ; out=i[j+timesteps]

    #append the index value of respective notes 
    x.append(list(map(lambda x:note2ind[x],inp)))
    y.append(note2ind[out])

x_new=np.array(x) 
y_new=np.array(y)

#reshape input and output for the model
x_new = np.reshape(x_new,(len(x_new),timesteps,1))
y_new = np.reshape(y_new,(-1,1))

#split the input and value into training and testing sets
#80% for training and 20% for testing sets
x_train,x_test,y_train,y_test = train_test_split(x_new,y_new,test_size=0.2,random_state=42)

#create the model
model = Sequential()
#create two stacked LSTM layer with the latent dimension of 256
model.add(LSTM(256,return_sequences=True,input_shape=(x_new.shape[1],x_new.shape[2])))
model.add(Dropout(0.2))
model.add(LSTM(256))
model.add(Dropout(0.2))
model.add(Dense(256,activation='relu'))

#fully connected layer for the output with softmax activation
model.add(Dense(len(note2ind),activation='softmax'))
model.summary()

#compile the model using Adam optimizer
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam',metrics=['accuracy'])

#train the model on training sets and validate on testing sets
model.fit(
    x_train,y_train,
    batch_size=128,epochs=80, 
    validation_data=(x_test,y_test))

#save the model for predictions
model.save("s2s")

#load the model
model=load_model("s2s")
#generate random index
index = np.random.randint(0,len(x_test)-1)
#get the data of generated index from x_test
music_pattern = x_test[index]

out_pred=[] #it will store predicted notes

#iterate till 200 note is generated
for i in range(200):

  #reshape the music pattern 
  music_pattern = music_pattern.reshape(1,len(music_pattern),1)
  
  #get the maximum probability value from the predicted output
  pred_index = np.argmax(model.predict(music_pattern))
  #get the note using predicted index and
  #append to the output prediction list
  out_pred.append(ind2note[pred_index])
  music_pattern = np.append(music_pattern,pred_index)
  
  #update the music pattern with one timestep ahead
  music_pattern = music_pattern[1:]

output_notes = []
for offset,pattern in enumerate(out_pred):
  #if pattern is a chord instance
  if ('.' in pattern) or pattern.isdigit():
    #split notes from the chord
    notes_in_chord = pattern.split('.')
    notes = []
    for current_note in notes_in_chord:
        i_curr_note=int(current_note)
        #cast the current note to Note object and
        #append the current note 
        new_note = note.Note(i_curr_note)
        new_note.storedInstrument = instrument.Piano()
        notes.append(new_note)
    
    #cast the current note to Chord object
    #offset will be 1 step ahead from the previous note
    #as it will prevent notes to stack up 
    new_chord = chord.Chord(notes)
    new_chord.offset = offset
    output_notes.append(new_chord)
  
  else:
    #cast the pattern to Note object apply the offset and 
    #append the note
    new_note = note.Note(pattern)
    new_note.offset = offset
    new_note.storedInstrument = instrument.Piano()
    output_notes.append(new_note)

#save the midi file 
midi_stream = stream.Stream(output_notes)
midi_stream.write('midi', fp='pred_music.mid')
