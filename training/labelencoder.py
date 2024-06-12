# Label Encoding Example
data_label = {
    'City': ['New York', 'Paris', 'Berlin', 'New York', 'Berlin']
}
df_label = pd.DataFrame(data_label)

# Applying LabelEncoder
label_encoder = LabelEncoder()
df_label['City_Label'] = label_encoder.fit_transform(df_label['City'])
print("Label Encoded DataFrame:\n", df_label)

# Ordinal Encoding Example
data_ordinal = {
    'Size': ['Small', 'Medium', 'Large', 'Medium', 'Small']
}
df_ordinal = pd.DataFrame(data_ordinal)

# Applying OrdinalEncoder
ordinal_encoder = OrdinalEncoder(categories=[['Small', 'Medium', 'Large']])
df_ordinal['Size_Ordinal'] = ordinal_encoder.fit_transform(df_ordinal[['Size']])
print("Ordinal Encoded DataFrame:\n", df_ordinal)