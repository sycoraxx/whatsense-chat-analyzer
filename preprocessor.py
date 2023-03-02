import re
import pandas as pd
import numpy as np
import datetime as dt
import emoji
import io

def startsWithDateAndTimeAndroid(s):
    pattern = '^([0-9]+)(\/)([0-9]+)(\/)([0-9]+), ([0-9]+):([0-9]+)(.)(AM|PM|am|pm)? -'
    result = re.match(pattern, s)
    if result:
        return True
    return False


def FindAuthor(s):
    s = s.split(":")
    if len(s) == 2:
        return True
    else:
        return False


def getDataPointAndroid(line):   
    splitLine = line.split(' - ') 
    dateTime = splitLine[0] 
    message = ' '.join(splitLine[1:])
    if FindAuthor(message): 
        splitMessage = message.split(':') 
        author = splitMessage[0] 
        message = ' '.join(splitMessage[1:])
    else:
        author = None
    return dateTime, author, message


def getDataPointIOS(line):   
    splitLine = line.split('] ') 
    dateTime = splitLine[0]
    if ',' in dateTime:
        date, time = dateTime.split(',') 
    else:
        date, time = dateTime.split(' ')
    message = ' '.join(splitLine[1:])
    if FindAuthor(message): 
        splitMessage = message.split(':') 
        author = splitMessage[0] 
        message = ' '.join(splitMessage[1:])
    else:
        author = None
    if time[5]==":":
        time = time[:5] + time[-3:]
    else:
        if 'AM' in time or 'PM' in time:
            time = time[:6] + time[-3:]
        else:
            time = time[:6]
    
    return date, time, author, message

def split_count(text):
    
    emoji_list = []
    emojis_iter = map(lambda y:y, emoji.UNICODE_EMOJI['en'].keys())
    regex_set = re.compile('|'.join(re.escape(em) for em in emojis_iter))
    emoji_list = regex_set.findall(text)
    return emoji_list

def preprocess(data):
    conversationPath = 'chat.txt' # chat file

    parsedData = [] 
    
    fp = io.StringIO(data)
    first = fp.readline()
    # if '[' in first:
        # device = 'ios'
    # else:
       # device = 'android'
    
    fp.readline()
    messageBuffer = []
    dateTime, author = None, None

    while True:
        line = fp.readline() 
        if not line: 
            break
        line = line.strip() 
        if startsWithDateAndTimeAndroid(line): 
            if len(messageBuffer) > 0: 
                parsedData.append([dateTime, author, ' '.join(messageBuffer)]) 
            messageBuffer.clear() 
            dateTime, author, message = getDataPointAndroid(line) 
            messageBuffer.append(message) 
        else:
            messageBuffer.append(line)
            

    df = pd.DataFrame(parsedData, columns=['DateTime', 'Author', 'Message']) # Initialising a pandas Dataframe.
    ### changing datatype of "Date" column.

    df["DateTime"] = df["DateTime"].str.strip()

    df.loc['DateTime'] = df.loc[:,'DateTime'].str.replace('\u202f', '')

    df["DateTime"] = pd.to_datetime(df["DateTime"], format='%d/%m/%Y, %I:%M %p', errors="ignore")
    df["DateTime"] = pd.to_datetime(df["DateTime"], format='%m/%d/%y, %I:%M %p', errors="ignore")

    no_author = df[df["Author"].isna()].index
    df.loc[no_author, 'Author'] = df.loc[no_author, 'Message'].str.split(': ', expand=True)[0]
    df.loc[no_author, 'Message'] = df.loc[no_author, 'Message'].str.split(': ', expand=True)[1]
    df = df.dropna(how='any') 

    
    df["Message"] = df["Message"].str.strip()
    df["Author"] = df["Author"].str.strip()

    df['only_date'] = df['DateTime'].dt.date
    df['year'] = df['DateTime'].dt.year
    df['month_num'] = df['DateTime'].dt.month
    df['month'] = df['DateTime'].dt.month_name()
    df['day'] = df['DateTime'].dt.day
    df['day_name'] = df['DateTime'].dt.day_name()
    df['hour'] = df['DateTime'].dt.hour
    df['minute'] = df['DateTime'].dt.minute

    period = []
    for hour in df[['day_name', 'hour']]['hour']:
        if hour == 23:
            period.append(str(hour) + "-" + str('00'))
        elif hour == 0:
            period.append(str('00') + "-" + str(hour + 1))
        else:
            period.append(str(hour) + "-" + str(hour + 1))

    df['period'] = period

    df = df[df['Message'] != '']
    return df