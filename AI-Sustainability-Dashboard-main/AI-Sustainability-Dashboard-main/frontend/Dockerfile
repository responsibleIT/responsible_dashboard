FROM node:22-alpine AS build
WORKDIR /app
COPY package*.json ./
RUN npm install --production=false
COPY . .
RUN npm run build --development

FROM nginx:1-alpine AS runtime
COPY --from=build /app/build /usr/share/nginx/html
COPY ./docker/nginx.conf /etc/nginx/nginx.conf