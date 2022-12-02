# Análise Estrutural - 2

Este repositório foi criado para documentação da disciplina de **Análise
Estrutural - 2**, que faz uso de scripts em Python para resolução de problemas
de engenharia.

## Como rodar

É indicado ter o Python em alguma versão > `3.8.X`

### Clonar o projeto

```sh
git clone https://github.com/zipluciano/analise-estrutural.git
```

### Acessar a raiz

```sh
cd ./analise-estrutural
```

### Criar ambiente virtual

<table>
  <tr>
    <td>
      Linux - bash | zsh
    </td>
    <td>
      conda create --name <envname> --file requirements.txt
    </td>
  </tr>
  <tr>
    <td>
      Windows - PowerShell
    </td>
    <td>
      conda create --name <envname> --file requirements.txt
    </td>
  </tr>
</table>

### Ativar ambiente virtual

<table>
  <tr>
    <td>
      Linux - bash | zsh
    </td>
    <td>
      conda activate <envname>
    </td>
  </tr>
  <tr>
    <td>
      Windows - PowerShell
    </td>
    <td>
      conda activate <envname>
    </td>
  </tr>
</table>

### Criar base de dados do Projeto - 01

<table>
  <tr>
    <td>
      Linux - bash | zsh
    </td>
    <td>
      python3 ./projeto_1_data.py
    </td>
  </tr>
  <tr>
    <td>
      Windows - PowerShell
    </td>
    <td>
      python .\projeto_1_data.py
    </td>
  </tr>
</table>

### Criar base de dados do Projeto - 02

<table>
  <tr>
    <td>
      Linux - bash | zsh
    </td>
    <td>
      python3 ./src/projeto-2/PerfisLista.py
    </td>
  </tr>
  <tr>
    <td>
      Windows - PowerShell
    </td>
    <td>
      python .\src\projeto-2\PerfisLista.py
    </td>
  </tr>
</table>

### Desativar ambiente virtual

```sh
conda deactivate
```
