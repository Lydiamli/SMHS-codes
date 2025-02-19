
# %% load modules
import pandas as pd
import numpy as np



# %% student
df_s = pd.read_spss("../data/raw/2014_SMHS_Share_Files/SMHS_Student_V1.sav")
df_s = df_s.filter(regex=f'^(?!{"s_"}).*$', axis=1) #remove repetitive variables
df_s['SA2Age'] = pd.to_numeric(df_s['SA2Age'], errors='coerce')
#count_greater_than_18 = (df_s['SA2Age'] > 18).sum() #156 students
df_s = df_s[(df_s['SA2Age'] <= 18) | df_s['SA2Age'].isna()]
df_s = df_s.rename(columns={"x_board_ID": "district", 
                            "x_idschool": "school", "x_class_ID": "class_id",
                            "x_student_ID": "student"})
df_s[["district", "school", "student", "class_id"]] = df_s[["district",
                                                            "school", "student", 
                                                            "class_id"]].fillna(0).astype(int)

df_s['district'] = df_s.district.apply(lambda x: "d" + str(x).zfill(2)) #45 unique district
df_s['school'] = df_s.school.apply(lambda x: "sc" + str(x).zfill(3)) #248 unique schools
df_s['class_id'] = df_s.class_id.apply(lambda x: str(x).zfill(6)) # 185 unique class_id
df_s['student'] = df_s.student.apply(lambda x: "s" + str(x).zfill(5))


#create proper ID variable 
df_s["teacher"] = df_s["district"]+df_s["school"] + df_s["class_id"] #1969 unique teacher
#df_s["teacher"] = df_s["teacher"].str.replace("sc", "t")

# define reorder columns
def reorder_column_function(data_frame):
    starting_cols_order = ['district', 'school', 'teacher', 'student']
    for col in data_frame.columns.values:
        if col not in starting_cols_order:
            starting_cols_order.append(col)
    return data_frame[starting_cols_order]
df_s = reorder_column_function(df_s)

df_s.loc[df_s["teacher"].apply(lambda x: x[-6:]=='000000'), 'teacher']= np.nan

# drop redudant variables
columns_to_drop = ['class_id', 'grade', "class"]
df_s.drop(columns = columns_to_drop, inplace = True)
# df_s.head()

# create proper variable names for student variables 
id_vars = ["district", "school", "teacher", "student"]
df_s.shape #(30968, 169)

# %% teacher 
df_t = pd.read_spss("../data/raw/2014_SMHS_Share_Files/SMHS_Teacher612_V1.sav")
df_t = df_t.filter(regex=f'^(?!{"t_"}).*$', axis=1) # remove redundant variables

df_t = df_t.rename(columns={'x_board_ID': 'district', 'x_idschool': 'school', 
                            'x_class_ID': 'class_id'})
df_t[["district","school", "class_id"]] = df_t[["district", "school",
                                                 "class_id"]].fillna(0).astype(int)


df_t['school'] = df_t.school.apply(lambda x: "sc" + str(x).zfill(3))
df_t['district'] = df_t.district.apply(lambda x: "d" + str(x).zfill(2))
df_t['class_id'] = df_t.class_id.apply(lambda x: str(x).zfill(6))
df_t["teacher"] = df_t["district"]+ df_t["school"] + df_t["class_id"] #1531 uniqe teacher
#df_t["teacher"] = df_t["teacher"].str.replace("sc", "t")
columns_to_drop = ['class_id', 'grade', "class"]
df_t.drop(columns = columns_to_drop, inplace = True)
df_t = pd.merge(df_s[['student','teacher']], df_t, on = 'teacher', how = 'outer')
# reorder columns
df_t = reorder_column_function(df_t)
# df_t.head()
df_t.shape #(31002, 137)

# %% principles
df_p = pd.read_spss("../data/raw/2014_SMHS_Share_Files/SMHS_Prinicipal_V1.sav")
df_p = df_p.filter(regex=f'^(?!{"p_"}).*$', axis=1) # remove redundant variables
df_p.drop(columns="PG2PrimaryRole", inplace=True)
df_p = df_p.rename(columns={'x_idschool': 'school', 'x_board_ID': 'district'})
df_p[["district","school"]] = df_p[["district", "school"]].fillna(0).astype(int)
df_p['school'] = df_p.school.apply(lambda x: "sc" + str(x).zfill(3)) #206 unique schools
df_p['district'] = df_p.district.apply(lambda x: "d" + str(x).zfill(2)) # 42 unique districts

df_p = pd.merge(df_s[["student","teacher", "school"]], 
                df_p, on = "school", how = "outer")
# reorder columns
df_p = reorder_column_function(df_p)
df_p.shape # (30969, 95)


# %% district
df_g = pd.read_spss("../data/raw/2014_SMHS_Share_Files/EXTERNAL_SMHS_SchoolAggregate_V2.sav")
df_g = df_g.rename(columns={'x_idschool': 'school', 'x_board_ID': 'district'})
 
df_g[["district","school"]] = df_g[["district", "school"]].astype(int)
df_g['school'] = df_g.school.apply(lambda x: "sc" + str(x).zfill(3))
df_g['district'] = df_g.district.apply(lambda x: "d" + str(x).zfill(2))

variables_to_keep = ["school", "district", "SCH_LVL_SEC",
                     "SCH_LANG_FR", "SCH_type_CATH", "School_Size_Enrol"]
df_g = df_g[variables_to_keep]

column_name_mapping = {
    "SCH_LANG_FR": "PSCH_language",
    "SCH_LVL_SEC": "PSCH_level",
    "SCH_type_CATH": "PSCH_type",
    "School_Size_Enrol": "PSCH_size"
}

df_g.rename(columns = column_name_mapping, inplace=True)

df_g = pd.merge(df_s[['student','teacher', "school"]], df_g, on='school', how ='outer')
df_g = reorder_column_function(df_g)
df_g.shape #(30968, 8)
# %% merge
merged_df = pd.merge(df_s, df_t, on = id_vars, how = "left")
merged_df = pd.merge(merged_df, df_p, on = id_vars, how = "left")
merged_df = pd.merge(merged_df, df_g, on = id_vars, how = "left")  #31124, 397
cleaned_df = merged_df.dropna(subset=['teacher']) #103 students without any teacher info 
cleaned_df = cleaned_df.drop(columns=["district", "school", "student"])
cleaned_df = cleaned_df.rename(columns={'teacher': 'cluster'})
cleaned_df.shape # (30865 394)
#%%
cleaned_df.to_csv("../data/raw/merged_smhs.csv", index = False)
# %%

