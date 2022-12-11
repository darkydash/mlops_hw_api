## ДЗ по MLPOPS Быкова К.В. aka Darky Dash (wtf wow!)

### Как запустить эту мишуру? (или команды чтоб я не забыл че делал когда протрезвею)

Сгенерить образ и залить на докерхаб (место с голыми образами -> hot hot hot!)
```bash
docker build -f main.Dockerfile -t darkydash/hse_mlservice_flask:0.0.1
docker login
docker image push darkydash/hse_mlservice_flask:0.0.1
```

Запустить компоуз
```bash
docker compose build
docker compose up -d
```

Проверить что работает
```bash
docker ps
```
