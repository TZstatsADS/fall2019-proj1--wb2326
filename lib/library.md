
#lyrics group by artist
def ofString(s):
    s= s.lower()
    s= s.replace('\n', ' ')
    s= s.replace(',', ' ')
    s= s.replace('\'', ' ')
    return s

def allTextByArtist(df):
    D={}
    for ind,val in df.iterrows():
        art = val["artist"]
        if art in D:
            D[art]= D[art]+ ofString(str(val["lyrics"]))
        else:
            D[art]= ofString(str(val["lyrics"]))
    return D

def drawCloud(s):
    wordclouddd = WordCloud(background_color="white",max_words=150,stopwords=stopwords).generate(s)
    #fig = plt.figure()
    #fig.set_figwidth(17)
    #fig.set_figheight(10)

    
    #plt.title('GG', color='#fafafa', size=30, y=1.01)
    #plt.annotate('GG', xy=(0, -.025), xycoords='axes fraction', fontsize=12, color='#fafafa')
    
    plt.imshow(wordclouddd)
    plt.axis("off")
    plt.figure()
    #plt.imshow(hcmask, cmap=plt.cm.gray)
    #plt.axis("off")
    #plt.show()
