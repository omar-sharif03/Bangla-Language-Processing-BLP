import os
import pickle

myFile = 'borno.txt'

#create list of vowels(both independent and dependent), consonants and compound characters
vowel_ind = ['অ', 'আ', 'ই', 'ঈ', 'উ', 'ঊ', 'ঋ', 'এ', 'ঐ', 'ও', 'ঔ']
vowel_dep = ['া', 'ি', 'ী', 'ু', 'ূ', 'ৃ', 'ে', 'ৈ', 'ো', 'ৌ', 'ং', '্']
consonant = ['ক','খ','গ','ঘ','ঙ',
             'চ','ছ','জ','ঝ','ঞ',
             'ট','ঠ','ড','ঢ','ণ',
             'ত','থ','দ','ধ','ন',
             'প','ফ','ব','ভ','ম',
             'য','র','ল','শ','ষ',
             'স','হ','ড়', 'ঢ়', 'য়',
             'ৎ']
jukto = ['ক্ক','ক্ট','ক্ট্র','ক্ত','ক্ত্র','ক্ব','ক্ম','ক্য','ক্র','ক্ল','ক্ষ','ক্ষ্ণ','ক্ষ্ম','ক্ষ্ম্য','ক্ষ্য','ক্স',
         'খ্য','খ্র',
         'গ্ণ', 'গ্ধ','গ্ধ্য','গ্ধ্র','গ্ন','গ্ন্য','গ্ব','গ্ম','গ্য','গ্র','গ্র্য','গ্ল',
         'ঘ্ন','ঘ্য','ঘ্র',
         'ঙ্ক','ঙ্ক্ষ','ঙ্খ','ঙ্গ', 'ঙ্ঘ', 'ঙ্ঘ্', 'ঙ্ম',
         'চ্চ', 'চ্ছ', 'চ্ছ্ব', 'চ্ছ্র', 'চ্ঞ', 'চ্ব', 'চ্য', 'চ্র',
         'জ্জ', 'জ্জ্ব', 'জ্ঝ', 'জ্ঞ', 'জ্ব', 'জ্য', 'জ্র',
         'ঞ্চ', 'ঞ্ছ', 'ঞ্জ', 'ঞ্ঝ',
         'ট্ট', 'ট্ব', 'ট্ম', 'ট্য', 'ট্র',
         'ড্ড', 'ড্ব' ,'ড্য', 'ড্র',
         'ড়্গ',
         'ঢ্য','ঢ্র',
         'ণ্ট', 'ণ্ঠ', 'ণ্ঠ্য', 'ণ্ড', 'ণ্ড্য', 'ণ্ড্র', 'ণ্ঢ', 'ণ্ণ', 'ণ্ব', 'ণ্ম', 'ণ্য',
         'ত্ত', 'ত্ত্য', 'ত্ত্ব', 'ত্থ', 'ত্ন', 'ত্ব', 'ত্ম', 'ত্ম্য', 'ত্য', 'ত্র', 'ত্র্য',
         'থ্ব', 'থ্য', 'থ্র',
         'দ্গ', 'দ্ঘ', 'দ্দ', 'দ্দ্ব', 'দ্ভ', 'দ্ভ্র', 'দ্ম', 'দ্য', 'দ্র', 'দ্র্য',
         'ধ্ন', 'ধ্ব', 'ধ্ম', 'ধ্য', 'ধ্র',
         'ন্ট', 'ন্ট্র', 'ন্ঠ', 'ন্ড', 'ন্ড্র', 'ন্ত', 'ন্ত্ব', 'ন্ত্য', 'ন্ত্র', 'ন্ত্র্য', 'ন্থ', 'ন্থ্র', 'ন্থ্র্য', 'ন্থ', 'ন্থ্র', 'ন্দ', 'ন্দ্য', 'ন্দ্ব', 'ন্দ্র', 'ন্ধ', 'ন্ধ্য', 'ন্ধ্ব', 'ন্ধ্র', 'ন্ন', 'ন্ব', 'ন্ম', 'ন্য',
         'প্ট','প্ত', 'প্ন', 'প্প', 'প্য', 'প্র', 'প্র্য','প্ল', 'প্স',
         'ফ্র', 'ফ্য', 'ফ্ল',
         'ব্জ', 'ব্দ', 'ব্ধ', 'ব্ব', 'ব্য', 'ব্র', 'ব্ল',
         'ভ্ব', 'ভ্য', 'ভ্র',
         'ম্ন', 'ম্প', 'ম্প্র', 'ম্ফ', 'ম্ব', 'ম্ব্র', 'ম্ভ', 'ম্ভ্র', 'ম্ম', 'ম্য', 'ম্র', 'ম্ল',
         'য্য',
         'র্ক', 'র্ক্য', 'র্গ্য', 'র্ঘ্য', 'র্চ্য', 'র্ছ্য', 'র্জ্য', 'র্ণ্য', 'র্ত্য', 'র্থ্য', 'র্ব্য', 'র্ম্য', 'র্শ্য', 'র্ষ্য', 'র্হ্য', 'র্খ', 'র্গ', 'র্গ্র', 'র্ঘ', 'র্চ', 'র্জ', 'র্ঝ', 'র্ট', 'র্ড', 'র্ণ', 'র্ত', 'র্ত্র', 'র্ধ', 'র্ধ্ব', 'র্ন', 'র্প', 'র্ফ', 'র্ভ', 'র্ম', 'র্য', 'র্ল', 'র্শ', 'র্শ্ব', 'র্ষ', 'র্স', 'র্হ', 'র্ঢ্য',
         'ল্ক', 'ল্ক্য', 'ল্গ', 'ল্ট', 'ল্ড', 'ল্প', 'ল্ফ', 'ল্ব', 'ল্ভ', 'ল্ম', 'ল্য', 'ল্ল',
         'শ্চ', 'শ্ছ', 'শ্ন', 'শ্ব', 'শ্ম', 'শ্য', 'শর', 'শ্ল',
         'ষ্ক', 'ষ্ক্র', 'ষ্ট', 'ষ্ট্য', 'ষ্ট্র', 'ষ্ঠ', 'ষ্ঠ্য', 'ষ্ণ', 'ষ্প', 'ষ্প্র', 'ষ্ফ', 'ষ্ম', 'স্ব', 'ষ্য',
         'স্ক', 'স্ক্র', 'স্খ', 'স্ট', 'স্ট্র', 'স্ত', 'স্ত্ব', 'স্ত্য', 'স্ত্র', 'স্থ', 'স্থ্য', 'স্ন', 'স্প', 'স্প্র', 'স্প্ল', 'স্ফ', 'স্ব', 'স্ম', 'স্য', 'স্র', 'স্ল',
         'হ্ণ', 'হ্ন', 'হ্ব', 'হ্ম', 'হ্য', 'হ্র', 'হ্ল'
        ]


#create an array which will contain all possible labels
borno =  [];

#include the independent vowels
borno.extend(vowel_ind);
#include dependent vowels
borno.extend(vowel_dep);
#include appropriate dependent vowels
#for i in range(len(vowel_dep)):
 #   if(i>=3 and i<=5):
  #      continue;
   # elif(i==8 or i==9):
    #    continue;
    #else:
     #   borno.append(vowel_dep[i]);

#include the consonants along with it's vowel modiifiers
#borno.extend(consonant);
#for c in consonant:
 #   s = "";
 #   s+=c;
 #   k = s;
 #   borno.append(k);
 #   k+=vowel_dep[3];
 #   borno.append(k);
 #   k = s;
 #   k+=vowel_dep[4];
 #   borno.append(k);
 #   k = s;
 #   k+=vowel_dep[5];
 #   borno.append(k);

#include consonants
borno.extend(consonant);
#print(borno);

#include the compound characters along with it's vowel modifiers
# for c in jukto:
#     s = "";
#     s+=c;
#     k = s;
#     borno.append(k);
#     k+=vowel_dep[3];
#     borno.append(k);
#     k = s;
#     k+=vowel_dep[4];
#     borno.append(k);
#     k = s;
#     k+=vowel_dep[5];
#     borno.append(k);


#save the array in a file
with open(myFile, 'wb') as f:
    pickle.dump(borno, f);

#restore the label array form file
with open(myFile , 'rb') as f:
    rs_borno = pickle.load(f);

#sort the array
rs_borno.sort();
#print(rs_borno);


#separeate word into characters
def get_labels(s):
    i = 0;
    l = len(s);

    labels = []; #for storing the label indexes for words

    ch = "";
    while(i<l):
        #print(s[i]);
        #print("ch = ", ch);

        if((s[i] in vowel_ind) or (s[i] in consonant)):
            if(len(ch)>0):
                #print(ch);
                for _ in ch:
                    labels.append(rs_borno.index(_));
                #labels.append(rs_borno.index(ch));
                ch = "";
            ch += s[i];
        elif(s[i] == '্'):
            ch += s[i];
            ch += s[i+1];
            #labels.append(rs_borno.index(s[i]));
            #labels.append(rs_borno.index(s[i+1]));
            i+=1;
        else:
            ind = vowel_dep.index(s[i]);
            if(ind>=3 and ind<=5):
                #ch += s[i];
                #print(ch);
                for _ in ch:
                    labels.append(rs_borno.index(_));
                labels.append(rs_borno.index(s[i]));
                ch = "";
            else:
                if(ind == 0 or ind == 2 or ind == 10):
                    #print(ch, s[i]);
                    for _ in ch:
                        labels.append(rs_borno.index(_));
                    labels.append(rs_borno.index(s[i]));
                    ch = "";
                elif(ind == 1 or ind == 6 or ind == 7):
                    #print(s[i], ch);
                    labels.append(rs_borno.index(s[i]));
                    for _ in ch:
                        labels.append(rs_borno.index(_));
                    ch = "";
                else:
                    if(ind == 8):
                        #print(vowel_dep[6], ch, vowel_dep[0]);
                        labels.append(rs_borno.index(vowel_dep[6]));
                        for _ in ch:
                            labels.append(rs_borno.index(_));
                        labels.append(rs_borno.index(vowel_dep[0]));
                    else:
                        #print(vowel_dep[6], ch, vowel_dep[2]);
                        labels.append(rs_borno.index(vowel_dep[6]));
                        for _ in ch:
                            labels.append(rs_borno.index(_));
                        labels.append(rs_borno.index(vowel_dep[2]));
                    ch = "";
        i+=1;

    if(len(ch) > 0 ):
        #print(ch);
        for _ in ch:
            labels.append(rs_borno.index(_));

    return labels;