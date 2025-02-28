
# Obtendo pesos para cada votante
aux = pd.DataFrame({'Votante': EE['Votante'].unique()}).\
    sort_values(by="Votante")
aux['Votacao'] = 1
aux['Peso'] = modelo.get_weights()[3]

# Retornando dados de ID para Descrição
descricao = f_aplica_PARA_DE(aux[['Votante','Votacao']],
                                  os.path.join(diretorio,'3.Auxiliares','D00. DE_PARA.csv'))
aux['Votante'] = descricao['Votante']
aux['Partido'] = [x[-5] for x in aux['Votante']]
aux.loc[aux['Votante'] == 'JOHNSON (D USA)','Partido'] = 'D'

# Gráfico com os pesos
import seaborn as sns
import matplotlib.pyplot as plt

def scatter_text(x, y, text_column, color_column, data, title, xlabel, ylabel):
    color_mapping = dict({"R": "red", "D": "blue"})
    # Create the scatter plot
    p1 = sns.scatterplot(x, y, hue=color_column, data=data, legend=True, palette=color_mapping)
    # Add text besides each point
    for line in range(0,data.shape[0]):
        partido = data[color_column][line]
        if(partido == 'R'):
            cor = 'red'
        else:
            cor = 'blue'
        p1.text(data[x][line]+0.01, data[y][line],
                data[text_column][line], horizontalalignment='left',
                size='medium', color=cor, weight='semibold')
    # Set title and axis labels
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    return p1

plt.figure(figsize=(15,15))
scatter_text('Peso', 'Peso', 'Votante', 'Partido',
             data = aux,
             title = 'Posição política dos candidatos na dimensão latente',
             xlabel = 'Dimensão Latente 1',
             ylabel = 'Dimensão Latente 1')
