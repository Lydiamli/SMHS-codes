rm(list = ls())
library(data.table)
library(tidyverse)
library(caret)

recode_variable <- function(data, variables, response_ordered) {
    data <- copy(data.table(data))
    data[, (variables) := lapply(.SD, function(x) as.numeric(factor(x, levels = response_ordered))), .SDcols = variables]
    return(data.table(data))
}

d1 <- fread("../data/clean/n_na.csv")
d1 <- d1[prop_na < 0.25]
variable_names <- d1[["variable"]] 
id_var <-  "cluster"
variable_names <- c(variable_names, id_var)


d0 <- fread("../data/raw/merged_smhs.csv")
d0 <- select(d0, all_of(variable_names))


# SA1: Student background information ####
#d0[, .N, SA1Sex]
resp <- c("Female?","Male?")
d0 <- recode_variable(d0, names(d0[, .(SA1Sex)]), resp)

#d0[, .N, SA3Grade]
d0[SA3Grade == "Other", SA3Grade := ""]
d0[, SA3Grade := as.double(SA3Grade)]

resp <- c("No", "Yes")
d0 <- recode_variable(d0, names(d0[, .(SA4Canadaborn)]), resp)

# d0[, .N, SA6Ab]
resp <- c("0.0", "Aboriginal/Native (e.g., First Nations, Métis, or Inuit)")
d0 <- recode_variable(d0, names(d0[, .(SA6Ab)]), resp)
#d0[, .N, SA6Black]
resp <- c("0.0", "Black African (e.g., Ghanaian, Kenyan), Black Caribbean (e.g., Jamaican, Haitian) or                             Black C")
d0 <- recode_variable(d0, names(d0[, .(SA6Black)]), resp)
#d0[, .N, SA6EA]
resp <- c("0.0", "East Asian (e.g., Chinese, Japanese, Korean)")
d0 <- recode_variable(d0, names(d0[, .(SA6EA)]), resp)
#d0[, .N, SA6Latin]
resp <- c("0.0", "Latin American, Central American, South American (e.g., Mexican, Colombian, Brazilian, Chilean)")
d0 <- recode_variable(d0, names(d0[, .(SA6Latin)]), resp)
#d0[, .N, SA6Other]
resp <- c("0.0", "Other")
d0 <- recode_variable(d0, names(d0[, .(SA6Other)]), resp)
#d0[, .N, SA6SA]
resp <- c("0.0", "South Asian (e.g., East Indian, Pakistani, Sri Lankan, Afghan, Bangladeshi)")
d0 <- recode_variable(d0, names(d0[, .(SA6SA)]), resp)
#d0[, .N, SA6WA]
resp <- c("0.0", "West Asian or Arab (e.g., Iraqi, Syrian, Lebanese, Egyptian)")
d0 <- recode_variable(d0, names(d0[, .(SA6WA)]), resp)
#d0[, .N, SA6White]
resp <- c("0.0", "White")
d0 <- recode_variable(d0, names(d0[, .(SA6White)]), resp)

#d0[, .N, SA6SEA]
resp <- c("0.0", "Southeast Asian (e.g., Vietnamese, Filipino, Cambodian, Malaysian, Laotian)")
d0 <- recode_variable(d0, names(d0[, .(SA6SEA)]), resp)

 # SB1: School climate ####
resp <- c("Disagree a LOT", "Disagree", 
          "Agree", "Agree a LOT")
d0 <- recode_variable(d0, names(select(d0, starts_with("SB1"))), resp)

# SB2: School belonging ####
resp <- c("Strongly disagree", "Disagree", 
          "Neither disagree nor agree", "Agree", "Strongly agree")
d0 <- recode_variable(d0, names(select(d0, starts_with("SB2"))), resp)

# SB3: School safety  ####
resp <- c("Not safe", "Somewhat safe", "Mostly safe", "Very safe")
d0 <- recode_variable(d0, names(select(d0, starts_with("SB3"))), resp)

#  SB4:Academic achievement  ####
resp <- c("D or lower (<60)", "C (60-69)", "B (70-79)", "A (80-90)")
d0 <- recode_variable(d0, names(select(d0, starts_with("SB4"))), resp)


# SB5: Extra-cirricular activites at school ####

resp <- c("Almost never", "About once a month",
          "About once a week",
          "A few times a week", "Most days")
d0 <- recode_variable(d0, names(select(d0, starts_with("SB5"))), resp)


# SB6: Bullying ####
resp <- c("Never", "Once or a few times",
          "Once or twice a month",
          "Once or twice a week", "Almost every day")
d0 <- recode_variable(d0, names(select(d0, starts_with("SB6"))), resp)

#  SB7: Truancy/Suspensions/Sent to Office ####
truancy_suspensions <- names(select(d0, starts_with("SB7")))
resp <- c("Never", "1 or 2 times",
          "3 or 4 times",
          "5 or more times")
d0 <- recode_variable(d0, truancy_suspensions, resp)

# SC1: Teaching Strategies and Interactions
resp <- c("Never", "Rarely",
          "Sometimes",
          "Often", "Always")
d0 <- recode_variable(d0, names(select(d0, starts_with("SC1"))), resp)

# SC2: Quality of student group interaction
resp <- c("Never", "Rarely",
          "Sometimes",
          "Often", "Always")
d0 <- recode_variable(d0, names(select(d0, starts_with("SC2"))), resp)

# SC3: Classroom Preparedness ####
# d0[, .N, keyby = SC3Attn] 
resp <- c("Never", "Once in a while",
          "About half the time",
          "Usually", "Always")
d0 <- recode_variable(d0, names(select(d0, starts_with("SC3"))), resp)
# d0[, .N, keyby = SC3Attn]

# SD1:Social Competence ####
# SD10Alc so cannot starts_with SD1
#d0[, .N, keyby = SD11]
resp <- c("Strongly disagree", "Disagree", "Agree", "Strongly agree")
d0 <- recode_variable(d0, names(d0[, .(SD11)]), resp)
d0 <- recode_variable(d0, names(d0[, .(SD12)]), resp)
d0 <- recode_variable(d0, names(d0[, .(SD13)]), resp)
d0 <- recode_variable(d0, names(d0[, .(SD14)]), resp)
d0 <- recode_variable(d0, names(d0[, .(SD15)]), resp)
d0 <- recode_variable(d0, names(d0[, .(SD16)]), resp)
d0 <- recode_variable(d0, names(d0[, .(SD17)]), resp)



# SD2: Positive mental health ####
resp <- c("Never", "Rarely",   "Sometimes",
          "Often", "Always")
d0 <- recode_variable(d0, names(select(d0, starts_with("SD2"))), resp)
# d0[, .N, keyby = SD2Happy]


# SD3: Friendship Quality ####
#d0[, .N, keyby = SD3Friend1]
resp <- c("Never or not true",
          "Sometimes or somewhat true",
          "Often or very true")
d0 <- recode_variable(d0,  names(select(d0, starts_with("SD3"))), resp)


# SD4: Healthy Lifestyle Behaviours ####
#d0[, .N, keyby = SD4Breakfast]
healthy_Lifestyle <- names(select(d0, starts_with("SD4")))
resp <- c("No days","1-2 days", "3-4 days", 
          "5-6 days",
          "Every day")
d0 <- recode_variable(d0, healthy_Lifestyle, resp)

# SD5: Mental health problems checklist ####
# d0[, .N, keyby = SD51]
mental_problems <- names(select(d0, starts_with("SD5")))
resp <- c("Never or       not true", "Sometimes or somewhat true", "Often or      very true")
d0 <- recode_variable(d0, mental_problems, resp)


# SD6-D7: Self-perception of emotional and behavioural problems and need for professional help ####
resp <- c("No ? Go to D8", "Yes")
d0 <- recode_variable(d0, names(d0[, .(SD6)]), resp)
#d0[, .N, keyby = SD6]


# no SD7

# SD8: Smoking cigarettes ####
#d0[, .N, keyby = SD8Smoke]
resp <- c("I usually smoke at least one cigarette a day.",
          "I used to smoke every day, but have not smoked a cigarette in the last month.",
          "I smoke sometimes, but not every day.", 
          "I have tried marijuana, but only once or twice.", 
          "I have never tried smoking, not even a few puffs."
          )
d0 <- recode_variable(d0, names(d0[, .(SD8Smoke)]), resp)


# SD9: Marijuana or Cannabis Product use ####
#d0[, .N, SD9Marij]
resp <- c("I usually smoke marijuana at least once a week or more.",
          "I smoke sometimes, but not every week.",
          "I used to smoke marijuana about once a week, but have not done so in the last month.", 
          "I have tried marijuana, but only once or twice.",
          "I have never tried marijuana." )
d0 <- recode_variable(d0, names(d0[, .(SD9Marij)]), resp)

# SD10: Alcohol Consumption ####
#d0[, .N, SD10Alc]
resp <- c("5 or more times", "4 times","3 times", "2 times", "Once", "Never")
d0 <- recode_variable(d0, names(d0[, .(SD10Alc)]), resp)


# SE1: Recieved mental health services at school ####
#d0[, .N, SE1]
resp <- c("No", "Yes")
d0 <- recode_variable(d0, names(d0[, .(SE1)]), resp)

# No SE2:Overall rating of help recieved

# SE3: Willingness to seek help ####
# d0[, .N, SE3]
resp <- c("No", "Yes ? Go to E5")
d0 <- recode_variable(d0, names(d0[, .(SE3)]), resp)

# No SE4

# SE5: Professional help seeking ####
# d0[, .N, SE5Doc] 
resp <- c("No", "Yes")
d0 <- recode_variable(d0, names(select(d0, starts_with("SE5"))), resp)

# SE6: Informal help seeking ####
#d0[, .N, SE6Relig] 
d0 <- recode_variable(d0, names(select(d0, starts_with("SE6"))), resp) 

# SF1: Family Structure ####
#d0[, .N, SF1BioMother]
resp <- c("0.0", "Biological mother")
d0 <- recode_variable(d0, names(d0[, .(SF1BioMother)]), resp)

#d0[, .N, SF1BioFather]
resp <- c("0.0", "Biological father")
d0 <- recode_variable(d0, names(d0[, .(SF1BioFather)]), resp)

#d0[, .N, SF1NonBioMother] # 1621
resp <- c("0.0", "Non-biological mother")
d0 <- recode_variable(d0, names(d0[, .(SF1NonBioMother)]), resp)

#d0[, .N, SF1NonBioFather]
resp <- c("0.0", "Non-biological father")
d0 <- recode_variable(d0, names(d0[, .(SF1NonBioFather)]), resp)

#d0[, .N, SF1OtherAdult]
resp <- c("0.0", "Other adult parent")
d0 <- recode_variable(d0, names(d0[, .(SF1OtherAdult)]), resp)

# d0[, .N, SF1Grandparent]
resp <- c("0.0", "Grandparent(s)")
d0 <- recode_variable(d0, names(d0[, .(SF1Grandparent)]), resp)

#d0[, .N, SF1OtherAdultRelative]
resp <- c("0.0", "Other adult relative(s)")
d0 <- recode_variable(d0, names(d0[, .(SF1OtherAdultRelative)]), resp)

#d0[, .N, SF1BrotherSister]
resp <- c("0.0", "Brother(s) or sister(s)")
d0 <- recode_variable(d0, names(d0[, .(SF1BrotherSister)]), resp)

#d0[, .N, SF1Other]
resp <- c("0.0", "Other(s)")
d0 <- recode_variable(d0, names(d0[, .(SF1Other)]), resp)

#d0[, .N, SF1Alone] # 310  
resp <- c("I live alone", "0.0")
d0 <- recode_variable(d0, names(d0[, .(SF1Alone)]), resp)

# SF2: Quality of Family Relationships ####
#d0[, .N, SF2ParentEnc]
family_rela <- names(select(d0, starts_with("SF2")))
resp <- c("Never", "Some of           the tim", "Most of             the time",
          "All of                 the time")
d0 <- recode_variable(d0, names(select(d0, starts_with("SF2"))), resp)


# SF3-F4: Family Origin (Immigration) ####
#d0[, .N, SF3MomCan]
resp <- c("No", "Yes")
d0 <- recode_variable(d0, names(d0[, .(SF3MomCan)]), resp)


#d0[, .N, SF4DadCan]
resp <- c("No", "Yes")
d0 <- recode_variable(d0, names(d0[, .(SF4DadCan)]), resp)


#d0[, .N, SF5ParentEd]
resp <- c("Did not graduate from high school", "Graduated high school",
          "Graduated college","Graduated university")
d0 <- recode_variable(d0, names(d0[, .(SF5ParentEd)]), resp)


# SF6:SES Indicator ####
resp <- c("No","Yes")
d0 <- recode_variable(d0, names(d0[, .(SF6Bedroom)]), resp)

family_asset <- names(select(d0, starts_with("SF7")))
resp <- c("None","1", "2", "3 or More")
d0 <- recode_variable(d0, family_asset, resp)


#SF8: Perception of survey language ####
#d0[, .N, SF8]
resp <- c("No difficulty at all",
          "Some difficulty", 
          "Moderate difficulty",
          "A lot of difficultly")
d0 <- recode_variable(d0, names(d0[, .(SF8)]), resp)



# Teacher survey ####
# TA1: School climate ####
#d0[, .N, TSSA11]
resp <- c("Disagree a LOT", "Disagree", "Agree", "Agree a LOT")
d0 <- recode_variable(d0, names(select(d0, starts_with("TSSA1"))), resp)

# TB1: Time spent as teacher of homeroom class  ####
#d0[, .N, TSSB1Week]  
resp <- c("1 week or less", "1-2 weeks", "2-3 weeks", "3-4 weeks", "4 weeks or more")
d0 <- recode_variable(d0, names(d0[, .(TSSB1Week)]), resp)


# TB2: Type of Class ####
# d0[, .N, TSSB2ClassType]
resp <- c("Regular classroom", "Special education classroom")
d0 <- recode_variable(d0, names(d0[, .(TSSB2ClassType)]), resp)


# TB3: Grade level of class (Teacher 6-12)####
resp <- c("0.0", "Grade 5")
d0 <- recode_variable(d0, names(d0[, .(TSSB3G5)]), resp)

resp <- c("0.0", "Grade 6")
d0 <- recode_variable(d0, names(d0[, .(TSSB3G6)]), resp)

resp <- c("0.0", "Grade 7")
d0 <- recode_variable(d0, names(d0[, .(TSSB3G7)]), resp)

resp <- c("0.0", "Grade 8")
d0 <- recode_variable(d0, names(d0[, .(TSSB3G8)]), resp)

resp <- c("0.0", "Grade 9")
d0 <- recode_variable(d0, names(d0[, .(TSSB3G9)]), resp)

resp <- c("0.0", "Grade 10")
d0 <- recode_variable(d0, names(d0[, .(TSSB3G10)]), resp)

resp <- c("0.0", "Grade 11")
d0 <- recode_variable(d0, names(d0[, .(TSSB3G11)]), resp)

resp <- c("0.0", "Grade 12")
d0 <- recode_variable(d0, names(d0[, .(TSSB3G12)]), resp)

resp <- c("0.0", "Other")
d0 <- recode_variable(d0, names(d0[, .(TSSB3Other)]), resp)


# TB4: Class Size####
#d0[, .N, TSSB4NoStudents]
resp <- c("1-15", "16-20", "21-25", "26-30", "31 or more")
d0 <- recode_variable(d0, names(d0[, .(TSSB4NoStudents)]), resp)


# TB5: Positive Behavioural Support ####
# d0[, .N, TSSB51]
resp <- c("Not at all", "Rarely",  "Sometimes", "Often", "Always")
d0 <- recode_variable(d0, names(select(d0, starts_with("TSSB5"))), resp)

# TB6: Disciplinary Approaches ####
#d0[, .N, TSSB6] 
resp <- c("0", "1",  "2", "3-5", "6 or more")
d0 <- recode_variable(d0, names(d0[, .(TSSB6)]), resp)

# TB7:Time Spent on Administrative Tasks, Keeping order, Teaching ###
#d0[, .N, TSSB7Admin]
resp <- c("More than 20%", "16-20%",  "11-15%","6-10%", "0-5%")
d0 <- recode_variable(d0, names(d0[, .(TSSB7Admin)]), resp)
d0 <- recode_variable(d0, names(d0[, .(TSSB7Order)]), resp)
#d0[, .N, TSSB7Teach]
resp <- c("0-25%", "26-50%",  "51-75%", "More than 75%")
d0 <- recode_variable(d0, names(d0[, .(TSSB7Teach)]), resp)

# TC1: Mental Health Promotion/Prevention Program ####
#d0[, .N, TSSC1SEL]
resp <- c("0.0", "Social and Emotional Learning (SEL) Program designed to foster positive emotional, behavioural, and interpersonal skills")
d0 <- recode_variable(d0, names(d0[, .(TSSC1SEL)]), resp)
#d0[, .N, TSSC1VPPP]
resp <- c("0.0", "Violence Prevention or Peace Promotion Program designed to make students and schools safer and more peaceful. Many addre")
d0 <- recode_variable(d0, names(d0[, .(TSSC1VPPP)]), resp)

#d0[, .N, TSSC1RPHPP]
resp <- c("0.0", "Risk Prevention or Health Promotion Program designed to reduce unhealthy behaviours, such as alcohol, tobacco or drug us")
d0 <- recode_variable(d0, names(d0[, .(TSSC1RPHPP)]), resp)

#d0[, .N, TSSC1ERSMCSP]
resp <- c("0.0", "Emotion Regulation, Stress Management or Coping Skills Programs designed to prevent or reduce problems with depression,")
d0 <- recode_variable(d0, names(d0[, .(TSSC1ERSMCSP)]), resp)

#d0[, .N, TSSC1Other]
resp <- c("0.0", "Other Program")
d0 <- recode_variable(d0, names(d0[, .(TSSC1Other)]), resp)

#d0[, .N, TSSC1NoPrograms]
resp <- c("No programs aimed at student mental health have been implemented in my classroom ? Go to Section D", "0.0")
d0 <- recode_variable(d0, names(d0[, .(TSSC1NoPrograms)]), resp)


# TD1: Classroom preparedness ####
#d0[, .N, TSSD1Ontime]
resp <- c("None", "Some", "About half", "Most", "Nearly all")
d0 <- recode_variable(d0, names(select(d0, starts_with("TSSD1"))), resp)


# TD2: Teaching strategies and interactions ####
resp <- c("Never", "Once or twice", "Weekly", "2 or 3 times a week", "Daily")
d0 <- recode_variable(d0, names(select(d0, starts_with("TSSD2"))), resp)
#d0[, .N, t_TSSD21]

# TD3: Evidence Based Practices ####
#d0[, .N, TSSD312]
resp <- c("Never", "Once or twice", "Weekly", "2 or 3 times a week", "Daily")
d0 <- recode_variable(d0, names(select(d0, starts_with("TSSD3"))), resp)



 # TE1: Personal Distress  ####
resp <- c("None of the time", "A little of the time", "Some of the time", "Most of the time", "All of the time")
d0 <- recode_variable(d0, names(select(d0, starts_with("TSSE1"))), resp)

# TF1: Barriers to Addressing Student Mental Health at school ####
resp <- c("Strongly disagree", "Somewhat disagree", "Somewhat agree", "Strongly agree")
d0 <- recode_variable(d0, names(select(d0, starts_with("TSSF1"))), resp)
 


# TG1: Resources and practices in place for mental health-related emergencies ####
resp <- c("Not in place", "Partially in place", "In place")
d0 <- recode_variable(d0, names(select(d0, starts_with("TSSG1"))), resp)


# TG1: Resources and practices in place for mental health-related emergencies ####
resp <- c("No", "Yes")
d0 <- recode_variable(d0, names(select(d0, starts_with("TSSH1"))), resp)


# TI1-I3: Teacher Demographics (Teacher 6-12)####
#d0[, .N,TSSI1Sex]
resp <- c("Female?", "Male?")
d0 <- recode_variable(d0, names(d0[, .(TSSI1Sex)]), resp)

#d0[, .N,TSSI2Canadaborn]
resp <- c("No", "Yes")
d0 <- recode_variable(d0, names(d0[, .(TSSI2Canadaborn)]), resp)

# TI3: Races ####
#d0[, .N, TSSI3White]
resp <- c("0.0", "White")
d0 <- recode_variable(d0, names(d0[, .(TSSI3White)]), resp)

#d0[, .N, TSSI3EA] # 643
resp <- c("0.0", "East Asian (e.g., Chinese, Japanese, Korean)")
d0 <- recode_variable(d0, names(d0[, .(TSSI3EA)]), resp)

#d0[, .N, TSSI3SEA] #123
resp <- c("0.0", "Southeast Asian (e.g., Vietnamese, Filipino, Cambodian, Malaysian, Laotian)")
d0 <- recode_variable(d0, names(d0[, .(TSSI3SEA)]), resp)

#d0[, .N, TSSI3SA]
resp <- c("0.0", "South Asian (e.g., East Indian, Pakistani, Sri Lankan, Afghan, Bangladeshi)")
d0 <- recode_variable(d0, names(d0[, .(TSSI3SA)]), resp)


#d0[, .N, TSSI3WA] #86
resp <- c("0.0", "West Asian (e.g., Iraqi, Syrian, Lebanese)") 
d0 <- recode_variable(d0, names(d0[, .(TSSI3WA)]), resp)


#d0[, .N, TSSI3Black] #536
resp <- c("0.0", "Black African (e.g., Ghanaian, Kenyan), Black Caribbean (e.g., Jamaican, Haitian) or                              Black")
d0 <- recode_variable(d0, names(d0[, .(TSSI3Black)]), resp)


#d0[, .N, TSSI3Latin] #336
resp <- c("0.0", "Latin American, Central American, or South American (e.g., Mexican, Colombian, Brazilian, Chilean)")
d0 <- recode_variable(d0, names(d0[, .(TSSI3Latin)]), resp)

#d0[, .N, TSSI3Arab] # 55
resp <- c("0.0", "Arab")
d0 <- recode_variable(d0, names(d0[, .(TSSI3Arab)]), resp)


#d0[, .N, TSSI3Ab] #253
resp <- c("0.0", "Aboriginal/Native (e.g.,  First Nations, Métis, or Inuit)")
d0 <- recode_variable(d0, names(d0[, .(TSSI3Ab)]), resp)


#d0[, .N, TSSI3Other] #857
resp <- c("0.0", "Other")
d0 <- recode_variable(d0, names(d0[, .(TSSI3Other)]), resp)


# TI4: Teaching Experience ####
#d0[, .N, TSSI4School]
resp <- c("Less than 1 year", "1-3 years", "3-5 years", "6-10 years", "Over 10 years")
d0 <- recode_variable(d0, names(d0[,.(TSSI4School)]), resp)

# Principal survey ####


# PA1: School Climate ####
#d0[, .N,  PA11]
resp <- c("Disagree a LOT", "Disagree", "Agree", "Agree a LOT")
d0 <- recode_variable(d0, names(select(d0, starts_with("PA1"))), resp)


# PB1: Social and Emotional Learning Programs ####
# d0[, .N,  PB1SEL]
resp <- c("No ? Go to B2", "Yes")
d0 <- recode_variable(d0, names(d0[, .(PB1SEL)]), resp)


# PB2: Violence Prevention or Peace Promotion Programs ####
# d0[, .N,  PB2VPPP]
resp <- c("No ? Go to B3", "Yes")
d0 <- recode_variable(d0, names(d0[, .(PB2VPPP)]), resp)


# PB3: Risk Prevention or Health Promotion Programs ####
# d0[, .N,  PB3RPHPP]
resp <- c("No ? Go to B4", "Yes")
d0 <- recode_variable(d0, names(d0[, .(PB3RPHPP)]), resp)


# PB4: Emotional Regulation Program ####
#d0[, .N,  PB4ERSMCSP]
resp <- c("No", "Yes")
d0 <- recode_variable(d0, names(d0[, .(PB4ERSMCSP)]), resp)


# PC1: Strategies to coordinate mental health activities and services ####
#d0[, .N, PC11]
resp <- c("Not applicable, no mental health staff at this school",
          "Not this school year", "Annually", "Quarterly", "Monthly")
d0 <- recode_variable(d0, names(select(d0, starts_with("PC1"))), resp)

# PC2: Formal agreements with children's mental health agency  ####
#d0[, .N, PC2] 
resp <- c("No", "Yes")
d0 <- recode_variable(d0, names(d0[, .(PC2)]), resp)


# PD1: How are mental health services staffed ####
#d0[, .N, PD1School]
resp <- c("0.0","Mental health staff are school-based (i.e., employees of the board or school who are assigned to this school and work on")
d0 <- recode_variable(d0, names(d0[, .(PD1School)]), resp)

#d0[, .N, PD1Board]
resp <- c("0.0","Mental health staff are board-based (i.e., employees of the board who are assigned to this school and travel to differen")
d0 <- recode_variable(d0, names(d0[, .(PD1Board)]), resp)

#d0[, .N, PD1Community]
resp <- c("0.0","Mental health staff are community-based (i.e., a community provider or organization with whom your school or board has a")
d0 <- recode_variable(d0, names(d0[, .(PD1Community)]), resp)

resp <- c("There are no mental health services available for students at this school.", "0.0")
d0 <- recode_variable(d0, names(d0[, .(PD1No)]), resp)

# No PD2

# PD3: Mental Health Services available at School ####
resp <- c("No", "Yes")
#d0[, .N, PD3DA] 
d0 <- recode_variable(d0, names(select(d0, starts_with("PD3"))), resp)

# PD4: Referral and Coordination Practices ####
#d0[, .N, PD4PassRef]
resp <- c("0.0","Staff make passive referrals (e.g. give brochures, lists, phone numbers of providers).")
d0 <- recode_variable(d0, names(d0[, .(PD4PassRef)]), resp)


#d0[, .N, PD4ActiveRef]
resp <- c("0.0","Staff make active referrals (e.g. staff complete form with family, make calls or appointments, assist with transportatio")
d0 <- recode_variable(d0, names(d0[, .(PD4ActiveRef)]), resp)

#d0[, .N, PD4FUS]
resp <- c("0.0","Staff follow-up with student/family (e.g. calls to ensure appointment kept, assess satisfaction with referral, need for")
d0 <- recode_variable(d0, names(d0[, .(PD4FUS)]), resp)

#d0[, .N, PD4FUP]
resp <- c("0.0","Staff follow-up with provider (via phone, e-mail, mail).")
d0 <- recode_variable(d0, names(d0[, .(PD4FUP)]), resp)

#d0[, .N, PD4TM]
resp <- c("0.0","Staff attend team meetings with community providers.")
d0 <- recode_variable(d0, names(d0[, .(PD4TM)]), resp)

#PD5: Resources and practices for mental health-related emergencies ####
#d0[, .N,  PD5Role]
resp <- c("Not in place", "Partially in place", "In place")
d0 <- recode_variable(d0, names(select(d0, starts_with("PD5"))), resp)


#PE1: Barriers to Addressing Student Mental Health at school ####
#d0[, .N, PE11]
resp <- c("Strongly disagree", "Somewhat disagree", "Somewhat agree", "Strongly agree")
d0 <- recode_variable(d0, names(select(d0, starts_with("PE1"))), resp)

# PF1: Professional Development Training ####
#d0[, .N, PF1Social]
resp <- c("No", "Yes")
d0 <- recode_variable(d0, names(select(d0, starts_with("PF1"))), resp)

# PG1: Demographics ####
#d0[, .N, PGSex]
resp <- c("Female?", "Male?")
d0 <- recode_variable(d0, names(d0[, .(PGSex)]), resp)


# PG3: Work Experience ####
resp <- c("Less than 1 year", "1-3 years", "3-5 years", "6-10 years", "Over 10 years")
d0 <- recode_variable(d0, names(d0[,.(PG3School)]), resp)


#d0[, .N,  PSCH_language]
resp <- c("english","french")
d0 <- recode_variable(d0, names(d0[, .(PSCH_language)]), resp)

resp <- c("elementary", "secondary")
d0 <- recode_variable(d0, names(d0[, .(PSCH_level)]), resp)

resp <- c("public", "catholic")
d0 <- recode_variable(d0, names(d0[, .(PSCH_type)]), resp)


# dummy code all binary categorical variables 
for (col_name in names(d0)) {
    unique_values <- unique(na.omit(d0[[col_name]]))
    if (length(unique_values) == 2 && all(unique_values %in% c(1, 2))) {
        d0[, (col_name) := as.integer(d0[[col_name]] - 1)]
    }
}


table(sapply(d0, typeof))

fwrite(d0, "../data/raw/encoded.csv")





