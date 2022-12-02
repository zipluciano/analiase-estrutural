# Análise Estrutural - 2

Este repositório foi criado para documentação da disciplina de **Análise
Estrutural - 2**, que faz uso de scripts em Python para resolução de problemas
de engenharia.

## Como rodar

É indicado ter o Python em alguma versão > `3.8.X`

### Clonar o projeto

```sh
git clone https://github.com/zipluciano/analise-estrutural-2.git
```

### Acessar a raiz

```sh
cd ./analise-estrutural-2
```

### Criar ambiente virtual

<table>
  <tr>
    <td>
      Linux - bash | zsh
    </td>
    <td>
      python3 -m venv ./.venv
    </td>
  </tr>
  <tr>
    <td>
      Windows - PowerShell
    </td>
    <td>
      python -m venv .\.venv
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
      source ./.venv/bin/activate
    </td>
  </tr>
  <tr>
    <td>
      Windows - PowerShell
    </td>
    <td>
      .\.venv\Scripts\Activate.ps1
    </td>
  </tr>
</table>

### Instalar pacotes

```sh
pip install -r ./requirements.txt
```

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
      python ./projeto_1_data.py
    </td>
  </tr>
</table>

### Desativar ambiente virtual

```sh
deactivate
```
